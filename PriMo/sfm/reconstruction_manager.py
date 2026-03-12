from mpsfm.baseclass import BaseClass
from mpsfm.sfm.mapper import MpsfmMapper


class ReconstructionManager(BaseClass):
    """Used to create the reconstruction object and manage the reconstruction process."""

    freeze_conf = False

    def _init(self, models=None):
        if models is None:
            models = {}
        self.models = models
        self.incremental_mapper = None

    def __call__(
        self,
        references,
        cache_dir,
        sfm_outputs_dir,
        scene_parser,
        scene="<custom>",
        extract_only=False,
        setup_only=False,
        **kwargs,
    ):

        exclude_init_pairs = set()
        print(50 * "=")
        if extract_only:
            print("\tSTARTING EXTRACTION")
            self.log(f"for {scene} and images {references} with imids {kwargs['ref_imids']}", level=1)
        else:
            print("\tSTARTING RECONSTRUCTION")
            self.log(f"for {scene} and images {references} with imids {kwargs['ref_imids']}", level=1)
        print(50 * "=")
        self.incremental_mapper = MpsfmMapper(
            conf=self.conf,
            references=references,
            cache_dir=cache_dir,
            sfm_outputs_dir=sfm_outputs_dir,
            scene=scene,
            scene_parser=scene_parser,
            models=self.models,
            extract_only=extract_only,
            setup_only=setup_only,
            **kwargs,
        )
        # check if has atribute extractor
        if hasattr(self.incremental_mapper, "extractor"):
            self.models = self.incremental_mapper.extractor.models
        elif hasattr(self.incremental_mapper, "models"):
            self.models = self.incremental_mapper.models

        if extract_only:
            print("Extraction complete")
            return None
        if setup_only:
            print("Reconstruction manager setup ready")
            return None
        mpsfm_rec, _ = self.incremental_mapper(
            refrec=scene_parser.rec, exclude_init_pairs=exclude_init_pairs, references=references
        )
        print(
            f"\nReconstrtuction complete with ({mpsfm_rec.num_reg_images()}/"
            f"{mpsfm_rec.num_images()}) registered images"
        )
        print(f"Rec has {mpsfm_rec.num_reg_images()}/{mpsfm_rec.num_images()} registered images")
        return mpsfm_rec
