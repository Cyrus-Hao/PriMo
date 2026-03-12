"""Module for optimizing depth maps."""

import time

import cv2
import numpy as np
import torch
from cholespy import CholeskySolverF, MatrixType
from tqdm import tqdm, trange

from mpsfm.sfm.scene.camera import CameraIntData
from mpsfm.utils.integration import move_bottom, move_left, move_right, move_top, setup_matrix_library, sigmoid

device_g = "cuda" if torch.cuda.is_available() else "cpu"
cp, csr_matrix, cg, identity, diags, sp = setup_matrix_library(device=device_g)


class IntVars:
    """Integration variables for depth map optimization."""

    Hessian = None
    wu = None
    wv = None
    A1 = None
    A2 = None
    A3 = None
    A4 = None
    energy_old = None
    integrated = False

    def move_to_device(self, device, move_hessian=False):
        """Move variables between numpy and cupy."""
        if device == "cpu":
            if device_g == "cuda":
                self.wu = cp.asnumpy(self.wu)
                self.wv = cp.asnumpy(self.wv)
                if hasattr(self.A1, "get"):  # only the case if on GPU
                    self.A1 = sp.csr_matrix(self.A1.get())
                    self.A2 = sp.csr_matrix(self.A2.get())
                    self.A3 = sp.csr_matrix(self.A3.get())
                    self.A4 = sp.csr_matrix(self.A4.get())
        else:
            self.wu = cp.asarray(self.wu)
            self.wv = cp.asarray(self.wv)
            self.A1 = csr_matrix(self.A1)
            self.A2 = csr_matrix(self.A2)
            self.A3 = csr_matrix(self.A3)
            self.A4 = csr_matrix(self.A4)


class IntegrationUncertainty:
    """Uncertainty propagation for normal integration."""

    def __init__(self, hessian, imshape, device="cuda"):
        self.n_rows = hessian.shape[0]
        data = torch.as_tensor(hessian.data, device=device)
        ii = torch.as_tensor(hessian.indptr, device=device)
        jj = torch.as_tensor(hessian.indices, device=device)
        self.solver = CholeskySolverF(self.n_rows, ii, jj, data, MatrixType.CSR)
        self.device = device
        self.imshape = imshape

    def solve(self, xy, verbose=False):
        """Solve the system of equations."""
        chunk_size = 128  # maximum batch size of cholespy
        if len(xy) > chunk_size:
            a = torch.cat(
                [self.solve(xy[i : i + chunk_size]) for i in (trange if verbose else range)(0, len(xy), chunk_size)], 0
            )
            return a

        xy = np.round(xy).astype(int)
        indices = np.ravel_multi_index(xy.T[::-1], self.imshape[:2])
        tgt = torch.zeros((self.n_rows, len(indices)), device=self.device)
        tgt[(indices, np.arange(len(indices)))] = 1
        x = torch.zeros_like(tgt)
        self.solver.solve(tgt, x)
        variances = (x).sum(0)
        return variances


class Integration(IntVars):
    """Integration class for depth map optimization."""

    def __init__(self):
        IntVars.__init__(self)
        self.device = device_g

        self.count_integrated = 0
        self.count_skipped = 0

    def _prepare_integration_variables(self) -> tuple[dict, bool]:
        variables = {}

        _, pts3dids, kps, depth3d, success = self.mpsfm_rec.project_image_3d_points(self.imid)
        if not success:
            print(f"Failed to project 3d points for {self.imid}")
            return None, False
        pts3dids = np.array(pts3dids)
        if self.conf.robust_triangles is not None:
            safe_mask = ~self.mpsfm_rec.find_points3D_with_small_triangulation_angle(
                min_angle=self.conf.robust_triangles, point3D_ids=pts3dids
            )
            pts3dids = pts3dids[safe_mask]
            kps = kps[safe_mask]
            depth3d = depth3d[safe_mask]
            self.log("Integration: Filtering", (~safe_mask).sum(), "out of ", len(safe_mask), level=3)
        kps *= np.array([self.camera.sx, self.camera.sy])
        kps = (kps + 0.5).astype(int)
        if len(pts3dids) == 0:
            zvars3d = np.array([])
            mask_canvas = slice(None)
        else:
            _, zvars3d = self.mpsfm_rec.point_covs.points_zvars(self.image, pts3dids)
            x, y = kps.T
            mask_canvas = (x >= 0) & (x < self.depth.data.shape[1]) & (y >= 0) & (y < self.depth.data.shape[0])

        variables["kps"] = kps[mask_canvas]
        variables["zvars3d"] = zvars3d[mask_canvas]
        variables["depth3d"] = depth3d[mask_canvas]

        # camera
        K = self.camera.calibration_matrix()
        variables["K"] = [
            K[1, 1] * self.camera.sy,
            K[0, 0] * self.camera.sx,
            K[1, 2] * self.camera.sy,
            K[0, 2] * self.camera.sx,
        ]

        return variables, True

    def integrate(self, cache_device="cpu"):
        """Integrate depth map from normals with depth constraints."""
        assert self.image.has_pose and self.depth.activated, "Image not registered or depth map not activated"
        variables, success = self._prepare_integration_variables()
        if (not success) or (variables is None):
            return False
        return self._integrate(cache_device=cache_device, **variables)

    def calc_energy(
        self,
        wu_plus,
        wu_minus,
        wv_plus,
        wv_minus,
        z,
        nx,
        ny,
        depth_precision,
        z_prior,
        sparse_precision,
        sparse_depth,
        sparse_ids,
    ):
        """Calculate energy for the optimization problem."""
        energy_matrix = (
            (wu_plus * (self.A1.dot(z) + nx) ** 2)
            + (wu_minus * (self.A2.dot(z) + nx) ** 2)
            + (wv_plus * (self.A3.dot(z) + ny) ** 2)
            + (wv_minus * (self.A4.dot(z) + ny) ** 2)
        )
        energy_matrix = cp.sum(energy_matrix)
        energy_matrix += cp.sum(self.conf.lambda1 * depth_precision * (z_prior - z) ** 2)
        if len(sparse_ids) > 0:
            energy_matrix += cp.sum(self.conf.lambda2 * sparse_precision * (sparse_depth - z[sparse_ids]) ** 2)
        return energy_matrix

    def calc_Amat(
        self,
        Nz,
        wu_plus,
        wu_minus,
        wv_plus,
        wv_minus,
        depth_precision,
        sparse_precision,
        sparse_ids,
        sparse_depth=True,
        camera=None,
    ):
        """Calculate the matrix A for the optimization problem."""
        camdata = self.camera if camera is None else camera
        data_term_top = wu_plus[camdata.has_top_mask_flat] * Nz["top_square"]
        data_term_bottom = wu_minus[camdata.has_bottom_mask_flat] * Nz["bottom_square"]
        data_term_left = wv_minus[camdata.has_left_mask_flat] * Nz["left_square"]
        data_term_right = wv_plus[camdata.has_right_mask_flat] * Nz["right_square"]

        # Initialize diagonal data term
        diagonal_data_term = cp.zeros(camdata.num_normals)

        # Add terms to diagonal data term
        diagonal_data_term[camdata.has_left_mask_flat] += data_term_left
        diagonal_data_term[camdata.has_left_mask_left_flat] += data_term_left
        diagonal_data_term[camdata.has_right_mask_flat] += data_term_right
        diagonal_data_term[camdata.has_right_mask_right_flat] += data_term_right
        diagonal_data_term[camdata.has_top_mask_flat] += data_term_top
        diagonal_data_term[camdata.has_top_mask_top_flat] += data_term_top
        diagonal_data_term[camdata.has_bottom_mask_flat] += data_term_bottom
        diagonal_data_term[camdata.has_bottom_mask_bottom_flat] += data_term_bottom

        # Add depth and sparse precision terms
        diagonal_data_term += self.conf.lambda1 * depth_precision
        if sparse_depth and len(sparse_ids) > 0:
            diagonal_data_term[sparse_ids] += self.conf.lambda2 * sparse_precision
        # Create diagonal matrix A_mat_d
        A_mat_d = csr_matrix(
            (diagonal_data_term, camdata.pixel_idx_flat, camdata.pixel_idx_flat_indptr),
            shape=(camdata.num_normals, camdata.num_normals),
        )

        # Create off-diagonal upper matrices
        A_mat_left_odu = csr_matrix(
            (-data_term_left, camdata.pixel_idx_left_center, camdata.pixel_idx_left_left_indptr),
            shape=(camdata.num_normals, camdata.num_normals),
        )
        A_mat_right_odu = csr_matrix(
            (-data_term_right, camdata.pixel_idx_right_right, camdata.pixel_idx_right_center_indptr),
            shape=(camdata.num_normals, camdata.num_normals),
        )
        A_mat_top_odu = csr_matrix(
            (-data_term_top, camdata.pixel_idx_top_center, camdata.pixel_idx_top_top_indptr),
            shape=(camdata.num_normals, camdata.num_normals),
        )
        A_mat_bottom_odu = csr_matrix(
            (-data_term_bottom, camdata.pixel_idx_bottom_bottom, camdata.pixel_idx_bottom_center_indptr),
            shape=(camdata.num_normals, camdata.num_normals),
        )

        # Sum off-diagonal upper matrices
        A_mat_odu = A_mat_top_odu + A_mat_bottom_odu + A_mat_right_odu + A_mat_left_odu

        # Compute final matrix A_mat
        A_mat = A_mat_d + A_mat_odu + A_mat_odu.T  # diagonal + upper triangle + lower triangle matrix

        return A_mat, diagonal_data_term

    def process_depth_prior(self, downscaled=False):
        """Process depth prior for integration."""
        data_prior = self.depth.data_prior
        uncertainty = self.depth.uncertainty

        if downscaled:
            H, W = data_prior.shape
            data_prior = cv2.resize(
                data_prior, (int(W // self.conf.downscale_factor), int(H // self.conf.downscale_factor))
            )
            uncertainty = cv2.resize(
                uncertainty, (int(W // self.conf.downscale_factor), int(H // self.conf.downscale_factor))
            )
        depth_prior = cp.asarray(data_prior, dtype=np.float64)
        depth_precision = self.conf.depth_magnitude_multiplier * cp.asarray(1 / (uncertainty + 1e-6), dtype=np.float64)
        depth_precision *= depth_prior**2  # log(d) -> var(log(d)) = var(d)/d^2
        z_prior = cp.log(depth_prior).flatten()
        depth_precision = depth_precision.flatten()
        valid_mask = self.depth.valid
        if downscaled:
            valid_mask = cv2.resize(
                valid_mask.astype(np.uint8),
                (int(W // self.conf.downscale_factor), int(H // self.conf.downscale_factor)),
            ).astype(bool)
        return depth_precision, z_prior, valid_mask

    def process_normals_prior(self, valid_mask, downscaled=False):
        """Process normals prior for integration."""
        if downscaled:
            normal_map = self.normals.data_downscaled
            normals_uncertainty = self.normals.uncertainty_downscaled
        else:
            normal_map = self.normals.data
            normals_uncertainty = self.normals.uncertainty
        normal_map = cp.asarray(normal_map, dtype=np.float64)
        nx = normal_map[..., 1].flatten()
        ny = normal_map[..., 0].flatten()
        nz = -normal_map[..., 2].flatten()

        normals_uncertainty[~valid_mask] = self.conf.large_number
        Vnx = (1 / self.conf.normals_magnitude_multiplier) * cp.asarray(normals_uncertainty[..., 1, 1].flatten())
        Vny = (1 / self.conf.normals_magnitude_multiplier) * cp.asarray(normals_uncertainty[..., 0, 0].flatten())
        Vnz = (1 / self.conf.normals_magnitude_multiplier) * cp.asarray(normals_uncertainty[..., 2, 2].flatten())
        return nx, ny, nz, Vnx, Vny, Vnz

    def process_sparse_depth(self, depth3d, zvars3d, kps):
        """Process sparse depth for integration."""
        sparse_ids = np.ravel_multi_index((kps[:, 1], kps[:, 0]), self.camera.nshape)
        depth3d = cp.asarray(depth3d)
        sparse_precision = cp.asarray(1 / zvars3d) * (depth3d**2)
        sparse_precision = sparse_precision.flatten()
        sparse_depth = cp.log(depth3d).flatten()
        return sparse_ids, sparse_precision, sparse_depth

    def load_depth_checkpoint(self, downscaled=False):
        """Load depth checkpoint for integration."""
        depth_init = self.depth.data
        if downscaled:
            H, W = depth_init.shape
            depth_init = cv2.resize(
                depth_init, (int(W // self.conf.downscale_factor), int(H // self.conf.downscale_factor))
            )
        depth_init = cp.asarray(depth_init)

        z = cp.log(cp.asarray(depth_init).flatten())
        return z

    def init_Nz(self, nz_u, nz_v, camdata=None):
        """Initialize the Nz dictionary for integration."""
        if camdata is None:
            camdata = self.camera
        Nz = {}
        Nz["left_square"] = nz_v[camdata.has_left_mask_flat] ** 2
        Nz["right_square"] = nz_v[camdata.has_right_mask_flat] ** 2
        Nz["top_square"] = nz_u[camdata.has_top_mask_flat] ** 2
        Nz["bottom_square"] = nz_u[camdata.has_bottom_mask_flat] ** 2
        return Nz

    def init_int_vars(
        self, z, fx, fy, cx, cy, nx, ny, nz, Vnx, Vny, Vnz, init=False, cache=True, return_all=False, camdata=None
    ):
        """Initialize integration variables."""
        if camdata is None:
            camdata = self.camera

        yy, xx = cp.meshgrid(cp.arange(camdata.nshape[1]), cp.arange(camdata.nshape[0]))
        xx = cp.flip(xx, axis=0)
        uu = xx.flatten() - cx
        vv = yy.flatten() - cy
        nz_u = uu * nx + vv * ny + fx * nz
        nz_v = uu * nx + vv * ny + fy * nz

        if init and self.integrated:
            A1, A2, A3, A4 = csr_matrix(self.A1), csr_matrix(self.A2), csr_matrix(self.A3), csr_matrix(self.A4)
        else:
            A3, A4, A1, A2 = self.generate_dx_dy(
                nz_horizontal=nz_v, nz_vertical=nz_u, step_size=self.conf.step_size, camdata=camdata
            )
        if cache:
            self.A3, self.A4, self.A1, self.A2 = A3, A4, A1, A2

        Nz = self.init_Nz(nz_u, nz_v, camdata=camdata)
        if init and self.integrated:
            self.wu = cp.asarray(self.wu)
            self.wv = cp.asarray(self.wv)
        else:
            wu, wv = self.update_W(z, A1=A1, A2=A2, A3=A3, A4=A4)
            if cache:
                self.wu, self.wv = wu, wv
        Duz_estim = -nx / nz_u
        Dvz_estim = -ny / nz_v
        ones_like_z = cp.ones_like(z)
        Nu_precision = 1 / (
            Vnx * ((uu * Duz_estim + ones_like_z) ** 2) + Vny * (vv * Duz_estim) ** 2 + fx**2 * Vnz * Duz_estim**2
        )
        Nv_precision = 1 / (
            Vnx * (uu * Dvz_estim) ** 2 + Vny * (vv * Dvz_estim + ones_like_z) ** 2 + fy**2 * Vnz * Dvz_estim**2
        )
        if return_all:
            return Nz, Nu_precision, Nv_precision, A1, A2, A3, A4, wu, wv
        return Nz, Nu_precision, Nv_precision

    def update_W(self, z, A1=None, A2=None, A3=None, A4=None):
        """Update the discontinuty weights for the optimization problem."""
        if A1 is None:
            A1, A2, A3, A4 = self.A1, self.A2, self.A3, self.A4
        wu = sigmoid((A2.dot(z)) ** 2 - (A1.dot(z)) ** 2, self.conf.k)  # top
        wv = sigmoid((A4.dot(z)) ** 2 - (A3.dot(z)) ** 2, self.conf.k)  # right
        return wu, wv

    def calc_Wpm(self, Nu_precision, Nv_precision, wu=None, wv=None):
        """Update the normal constraint weights."""
        if wu is None:
            wu = self.wu
        if wv is None:
            wv = self.wv
        wu_plus = wu * Nu_precision
        wu_minus = (1 - wu) * Nu_precision
        wv_plus = wv * Nv_precision
        wv_minus = (1 - wv) * Nv_precision
        return wu_plus, wu_minus, wv_plus, wv_minus

    def should_refine(self, energy):
        """Check if the energy has changed significantly."""
        relative_energy = cp.abs(energy - self.energy_old) / self.energy_old
        return relative_energy > self.conf.tol

    def _integrate(self, depth3d, zvars3d, kps, K, cache_device="cpu", init=True):
        fx, fy, cx, cy = K

        depth_precision, z_prior, valid_mask = self.process_depth_prior()
        nx, ny, nz, Vnx, Vny, Vnz = self.process_normals_prior(valid_mask)
        z = self.load_depth_checkpoint()
        sparse_ids, sparse_precision, sparse_depth = self.process_sparse_depth(depth3d, zvars3d, kps)

        if self.conf.scale_filter:
            scale_factor = self.conf.scale_filter_factor
            div = cp.exp(sparse_depth) / cp.exp(z_prior[sparse_ids])
            valid = (div < scale_factor) * (div > (1 / scale_factor))
            sparse_ids = sparse_ids[valid.get() if device_g == "cuda" else valid]
            sparse_precision = sparse_precision[valid]
            sparse_depth = sparse_depth[valid]

        Nz, Nu_precision, Nv_precision = self.init_int_vars(
            z,
            fx,
            fy,
            cx,
            cy,
            nx,
            ny,
            nz,
            Vnx,
            Vny,
            Vnz,
            init=init,
        )
        wu_plus, wu_minus, wv_plus, wv_minus = self.calc_Wpm(Nu_precision, Nv_precision)
        energy = self.calc_energy(
            wu_plus,
            wu_minus,
            wv_plus,
            wv_minus,
            z,
            nx,
            ny,
            depth_precision,
            z_prior,
            sparse_precision,
            sparse_depth,
            sparse_ids,
        )
        tic = time.time()

        if self.integrated and not self.should_refine(energy):
            if self.conf.verbose > 2:
                print("Energy hasn't changed. Skipping this frame.")
            self.count_integrated += 1
            return False

        self.count_skipped += 1
        energy_0 = min_energy = energy
        vis_energy = [cp.asnumpy(energy)]

        pbar = tqdm(range(self.conf.max_iter)) if self.conf.verbose > 1 else range(self.conf.max_iter)
        for i in pbar:
            A_mat, diagonal_data_term = self.calc_Amat(
                Nz, wu_plus, wu_minus, wv_plus, wv_minus, depth_precision, sparse_precision, sparse_ids
            )
            b_vec = (
                self.A1.T @ (wu_plus * (-nx))
                + self.A2.T @ (wu_minus * (-nx))
                + self.A3.T @ (wv_plus * (-ny))
                + self.A4.T @ (wv_minus * (-ny))
            )
            b_vec += self.conf.lambda1 * depth_precision * z_prior
            if len(sparse_ids) > 0:
                b_vec[sparse_ids] += self.conf.lambda2 * sparse_precision * sparse_depth

            D = csr_matrix(
                (
                    1 / cp.clip(diagonal_data_term, 1e-5, None),
                    self.camera.pixel_idx_flat,
                    self.camera.pixel_idx_flat_indptr,
                ),
                shape=(self.camera.num_normals, self.camera.num_normals),
            )
            if "rtol" in cg.__code__.co_varnames:  # supporting different versions...
                z, _ = cg(A_mat, b_vec, x0=z, M=D, maxiter=self.conf.cg_max_iter, rtol=self.conf.cg_tol)
            else:
                z, _ = cg(A_mat, b_vec, x0=z, M=D, maxiter=self.conf.cg_max_iter, tol=self.conf.cg_tol)

            # Update weights
            self.wu, self.wv = self.update_W(z)
            wu_plus, wu_minus, wv_plus, wv_minus = self.calc_Wpm(Nu_precision, Nv_precision)

            # Check for convergence
            energy_old = energy
            min_energy = min(energy, min_energy)

            energy = self.calc_energy(
                wu_plus,
                wu_minus,
                wv_plus,
                wv_minus,
                z,
                nx,
                ny,
                depth_precision,
                z_prior,
                sparse_precision,
                sparse_depth,
                sparse_ids,
            )
            vis_energy.append(cp.asnumpy(energy))

            relative_energy = cp.abs(energy - energy_old) / energy_old
            relative_energy_min = cp.abs(energy - min_energy) / min_energy
            if self.conf.verbose > 2:
                pbar.set_description(
                    f"step {i + 1}/{self.conf.max_iter} energy: {energy:.3e}"
                    f" relative energy: {relative_energy:.3e}"
                )
            if (
                (relative_energy < self.conf.tol and (energy_old - energy) > 0)
                or (relative_energy_min < self.conf.tol and (min_energy - energy) > 0)
            ) and energy < energy_0:
                break
            if energy > energy_0:
                self.integrated = True
                self.energy_old = energy_0
                self.log("Energy increased. Probably too noisy???. Skipping this frame.", level=2)
                return False
        toc = time.time()

        if self.conf.verbose > 1:
            print(f"Total time: {toc - tic:.3f} sec")
            print(f"Energy: {energy_0} --> {energy}.  Steps: {i+1}")

        self.depth.data = cp.asnumpy(cp.exp(z.reshape(self.camera.nshape))).astype(np.float64)
        self.integrated = True

        self.energy_old = energy
        self.move_to_device(cache_device)
        return True

    def calculate_hessian(self, downscaled, ignore_depths=None):
        """Calculate the Hessian matrix for the optimization problem."""
        camdata = self.camera
        if ignore_depths is None:
            ignore_depths = self.conf.ignore_depths
        if downscaled:
            camdata = CameraIntData(
                int(camdata.int_height // self.conf.downscale_factor),
                int(camdata.int_width // self.conf.downscale_factor),
            )

        kwargs, _ = self._prepare_integration_variables()
        depth3d, zvars3d, kps = kwargs["depth3d"], kwargs["zvars3d"], kwargs["kps"]
        if downscaled:
            kps = kps // self.conf.downscale_factor
            kps = kps.astype(int)
            fx, fy, cx, cy = [val / self.conf.downscale_factor for val in kwargs["K"]]
        else:
            fx, fy, cx, cy = kwargs["K"]
        z = self.load_depth_checkpoint(downscaled=downscaled)
        sparse_ids, sparse_precision, _ = self.process_sparse_depth(depth3d, zvars3d, kps)
        depth_precision, _, valid_mask = self.process_depth_prior(downscaled=downscaled)
        nx, ny, nz, Vnx, Vny, Vnz = self.process_normals_prior(valid_mask, downscaled=downscaled)
        Nz, Nu_precision, Nv_precision, _, _, _, _, wu, wv = self.init_int_vars(
            z, fx, fy, cx, cy, nx, ny, nz, Vnx, Vny, Vnz, init=False, cache=False, return_all=True, camdata=camdata
        )
        wu_plus, wu_minus, wv_plus, wv_minus = self.calc_Wpm(Nu_precision, Nv_precision, wu=wu, wv=wv)
        if ignore_depths:
            self.Hessian, _ = self.calc_Amat(
                Nz,
                wu_plus,
                wu_minus,
                wv_plus,
                wv_minus,
                depth_precision,
                sparse_precision,
                sparse_ids,
                sparse_depth=False,
                camera=camdata,
            )
        else:
            self.Hessian, _ = self.calc_Amat(
                Nz,
                wu_plus,
                wu_minus,
                wv_plus,
                wv_minus,
                depth_precision,
                sparse_precision,
                sparse_ids,
                sparse_depth=True,
                camera=camdata,
            )

    def calculate_int_covs_at_points(self, pts, verbose=False, Hessian=None, downscaled=None, ignore_depths=None):
        """Calculates integrated depth covariance at points."""
        if downscaled is None:
            downscaled = self.conf.downscaled
        if downscaled:
            if Hessian is None:
                if self.Hessian is None:
                    self.calculate_hessian(downscaled=downscaled, ignore_depths=ignore_depths)
                Hessian = self.Hessian
            data_shape = self.normals.data_downscaled.shape
            kps = pts // self.conf.downscale_factor
        else:
            if Hessian is None:
                if self.Hessian is None:
                    self.calculate_hessian(downscaled=downscaled, ignore_depths=ignore_depths)
                Hessian = self.Hessian
            data_shape = self.depth.data.shape
            kps = pts
        solver = IntegrationUncertainty(Hessian, data_shape, device=self.device)

        varlogd = solver.solve(kps, verbose=verbose)

        try:
            varlogd = varlogd.cpu().numpy()
        except Exception:
            varlogd = varlogd.get()
        return varlogd

    def calculate_int_covs_at_kps(self, Hessian=None, pts2d=None, downscaled=None):
        """Calculates propagated depth covariance at keypoints"""
        kps = self.mpsfm_rec.keypoints(self.imid)
        if pts2d is None:
            pts2d = np.arange(len(kps))
        else:
            kps = kps[pts2d]

        kps_down = kps * np.array([self.camera.sx, self.camera.sy])
        log_uncert = self.calculate_int_covs_at_points(kps_down, Hessian=Hessian, downscaled=downscaled)
        uncert = log_uncert * self.depth.data_prior_at_kps(kps) ** 2  # var(log(d)) = var(d)/d^2
        self.depth.uncertainty_update[pts2d] = uncert
        return uncert

    def calculate_int_covs_for_entire_image(self, downscaled=None, ignore_depths=False):
        """Calculates propagated depth covariance for the entire image."""
        xx, yy = np.meshgrid(np.arange(self.camera.nshape[1]), np.arange(self.camera.nshape[0]))
        return (
            self.calculate_int_covs_at_points(
                np.array([xx.flatten(), yy.flatten()]).T,
                verbose=self.conf.verbose > 0,
                downscaled=downscaled,
                ignore_depths=ignore_depths,
            ).reshape(self.camera.nshape)
            * self.depth.data**2
        )

    def generate_dx_dy(self, nz_horizontal, nz_vertical, step_size=1, camdata=None):
        """Generate the D matrices for the optimization problem."""
        # pixel coordinates
        # ^ vertical positive
        # |
        # |
        # |
        # o ---> horizontal positive
        if camdata is None:
            camdata = self.camera
        num_pixel = camdata.num_normals

        # Generate an integer index array with the same shape as the mask.
        pixel_idx = cp.zeros(camdata.nshape, dtype=int)
        pixel_idx[...] = cp.arange(pixel_idx.size).reshape(pixel_idx.shape)

        # Extract the horizontal and vertical components of the normal vectors for the neighboring pixels.
        nz_left = nz_horizontal[camdata.has_left_mask.flatten()]
        nz_right = nz_horizontal[camdata.has_right_mask.flatten()]
        nz_top = nz_vertical[camdata.has_top_mask.flatten()]
        nz_bottom = nz_vertical[camdata.has_bottom_mask.flatten()]

        data = cp.stack([-nz_left / step_size, nz_left / step_size], -1).flatten()

        indices = cp.stack(
            (pixel_idx[move_left(camdata.has_left_mask)], pixel_idx[camdata.has_left_mask]), -1
        ).flatten()
        indptr = cp.concatenate([cp.array([0]), cp.cumsum(camdata.has_left_mask.flatten().astype(int) * 2)])
        D_horizontal_neg = csr_matrix((data, indices, indptr), shape=(num_pixel, num_pixel))

        data = cp.stack([-nz_right / step_size, nz_right / step_size], -1).flatten()
        indices = cp.stack(
            (pixel_idx[camdata.has_right_mask], pixel_idx[move_right(camdata.has_right_mask)]), -1
        ).flatten()
        indptr = cp.concatenate([cp.array([0]), cp.cumsum(camdata.has_right_mask.flatten().astype(int) * 2)])
        D_horizontal_pos = csr_matrix((data, indices, indptr), shape=(num_pixel, num_pixel))

        data = cp.stack([-nz_top / step_size, nz_top / step_size], -1).flatten()
        indices = cp.stack((pixel_idx[camdata.has_top_mask], pixel_idx[move_top(camdata.has_top_mask)]), -1).flatten()
        indptr = cp.concatenate([cp.array([0]), cp.cumsum(camdata.has_top_mask.flatten().astype(int) * 2)])
        D_vertical_pos = csr_matrix((data, indices, indptr), shape=(num_pixel, num_pixel))

        data = cp.stack([-nz_bottom / step_size, nz_bottom / step_size], -1).flatten()
        indices = cp.stack(
            (pixel_idx[move_bottom(camdata.has_bottom_mask)], pixel_idx[camdata.has_bottom_mask]), -1
        ).flatten()
        indptr = cp.concatenate([cp.array([0]), cp.cumsum(camdata.has_bottom_mask.flatten().astype(int) * 2)])
        D_vertical_neg = csr_matrix((data, indices, indptr), shape=(num_pixel, num_pixel))

        return D_horizontal_pos, D_horizontal_neg, D_vertical_pos, D_vertical_neg
