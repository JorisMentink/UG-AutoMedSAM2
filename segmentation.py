from DataLoader import DataLoader
from sam2.build_sam import build_sam2_video_predictor_npz
from segmentation_util import run_medsam2_inference_from_arrays, compile_prompt_sets
from scipy.ndimage import distance_transform_edt
import numpy as np

class Segmentation:
    def __init__(
        self,
        data: DataLoader,
        checkpoint="checkpoints/MedSAM2_latest.pt",
        cfg="configs/sam2.1_hiera_t512.yaml",
        use_mask_input_as_output_without_sam=True,
        pred_obj_scores=True
    ):
        """
        Instantializes the segmentation handler.
        
        INPUTS:
        data: DataLoader instance containing the loaded data.
        checkpoint: Path to the SAM2 checkpoint file.
        cfg: Path to the SAM2 configuration file.
        use_mask_input_as_output_without_sam: If True, disables mask input as output, leading to continuous logits instead of -10/10.
        pred_obj_scores: If True, enables object presence predictor, which clips logits to -1024 when the model thinks no object is present.
        
        OUTPUTS:
        Initializes the Segmentation instance with the provided data and SAM2 predictor.
        """

        
        if not isinstance(data, DataLoader):
            raise TypeError(
                "Segmentation expects a DataLoader instance. "
                "Please create one first: data = DataLoader(...)"
            )

        self.data = data

        self.parentfolder = data.parentfolder
        self.subject_nr = data.subject_nr
        self.subject_name = data.subject_name
        self.volume_of_interest = data.volume_of_interest
        self.verbose = data.verbose


        self.img = data.img
        self.mask = data.mask
        self.gt = data.gt
        self.img_spacing = data.img_spacing

        self.prompt_dict_list = []
        self.prompts_by_slice = {}

        if self.verbose:
            print(
                f"Initialized Segmentation for subject {self.subject_nr} "
                f"with volume of interest '{self.volume_of_interest}'"
            )
            print(f"Mask shape: {self.mask.shape}")
            print(f"Building SAM predictor from checkpoint: {checkpoint}")

        self.predictor = build_sam2_video_predictor_npz(cfg, checkpoint)

        #Disables mask input as output, which leads to continuous logits instead of -10/10, but was found to not be relevant
        self.predictor.use_mask_input_as_output_without_sam = use_mask_input_as_output_without_sam
        #Disables object presence predictor, disables the -1024 clipping when model thinks no object is present.
        #This disabling was excluded due to noisy recontours. NO_OBJ_SCORE was set to -10 instead (conversion inside segmentation.py, not in sam2_base)
        self.predictor.pred_obj_scores = pred_obj_scores

        if self.verbose:
            print(
                "use_mask_input_as_output_without_sam:",
                self.predictor.use_mask_input_as_output_without_sam
            )

    def reduce_to_slice(self, selected_slice):
        "Function that reduces the 3D volume to a single slice for specific low-computational cost testing."

        #This way of indexing maintains 3d nature necessary for SAM2 
        self.img = self.img[selected_slice:selected_slice+1]
        self.mask = self.mask[selected_slice:selected_slice+1]
        self.gt = self.gt[selected_slice:selected_slice+1]

        if self.compiled_prompt_sets is not None:
            for set_name, (prompt_dict, weight) in self.compiled_prompt_sets.items():
                self.compiled_prompt_sets[set_name] = ({0: prompt_dict[selected_slice]}, weight)


    def load_dense_prompt(self):
        "Function that loads existing dense mask as prompt"

        if self.mask is None or self.mask.sum() == 0:
            raise ValueError("No valid mask found to load as prompt.")

        dense_prompts = {}

        for z in range(self.mask.shape[0]):
            mask_2d = self.mask[z].astype(np.uint8)

            if mask_2d.sum() == 0:
                continue

            dense_prompts[z] = {
                "points": None,
                "point_labels": None,
                "bbox": None,
                "mask_input": mask_2d,
            }

        self.dense_prompt_set = dense_prompts

        return self.dense_prompt_set

    def load_gaussianized_dense_prompt(
        self,
        strength=30.0,
        gamma=4.0,
    ):
        """
        Load the existing binary mask as a SAMRefiner-style Gaussian mask prompt.

        Background values are -strength.
        Foreground values range continuously from near 0 to +strength,
        with the strongest values furthest from the mask boundary.
        """

        if self.mask is None or not np.any(self.mask):
            raise ValueError("No valid mask found to load as prompt.")

        if strength <= 0:
            raise ValueError("strength must be greater than zero.")

        if gamma <= 0:
            raise ValueError("gamma must be greater than zero.")

        gaussian_prompts = {}

        self.gaussian_mask_logits = np.full(
            self.mask.shape,
            -strength,
            dtype=np.float32,
        )

        for z in range(self.mask.shape[0]):
            mask_2d = self.mask[z].astype(bool)

            if not mask_2d.any():
                continue

            #Distance from each foreground pixel to the mask boundary
            distance = distance_transform_edt(mask_2d).astype(np.float32)

            max_distance = distance.max()
            mask_area = max(mask_2d.sum() / gamma, 1.0)

            #SAMRefiner Gaussian foreground weighting
            gaussian_weight = np.exp(
                -((distance - max_distance) ** 2) / mask_area
            )

            #Background remains -strength
            mask_logits = np.full(
                mask_2d.shape,
                -strength,
                dtype=np.float32,
            )

            #Foreground ranges from near 0 to +strength
            mask_logits[mask_2d] = (
                strength * gaussian_weight[mask_2d]
            )

            self.gaussian_mask_logits[z] = mask_logits

            gaussian_prompts[z] = {
                "points": None,
                "point_labels": None,
                "bbox": None,
                "mask_input": mask_logits,
            }

        self.gaussian_dense_prompt_set = gaussian_prompts

        return self.gaussian_dense_prompt_set

    def check_loaded_prompts(self):
        """
        Prints summary of loaded prompts, including number of slices with prompts and their range.
        """

        print("\nPrompt summary")
        print("-" * 40)
        if hasattr(self, "compiled_prompt_sets") and self.compiled_prompt_sets is not None:
            print(f"Compiled prompt sets: {len(self.compiled_prompt_sets)}")

            for set_name, (prompt_dict, weight) in self.compiled_prompt_sets.items():
                slices = sorted(prompt_dict.keys())

                if len(slices) > 0:
                    print(f"  {set_name}: {len(slices)} slices "
                        f"({slices[0]} to {slices[-1]}), weight={weight}")
                else:
                    print(f"  {set_name}: 0 slices, weight={weight}")

        else:
            print("Compiled prompt sets: not created")

        print("-" * 40)


    def compile_prompt_sets(self, prompt_dict_list=None, prompt_set_names=None, prompt_set_weights=None):
        """
        Compiles prompt set dictionaries to be used for segmentation. If no prompt dict list is provided, this function defaults to use the dense mask as the only prompt set.
        
        INPUTS:
        prompt_dict_list: List of dictionaries, each containing prompts for a specific set. Each dictionary should have slice indices as keys and prompt data as values.
        prompt_set_names: List of names corresponding to each prompt set in prompt_dict_list.
        prompt_set_weights: List of weights corresponding to each prompt set in prompt_dict_list. The sum of weights should be 1.0.

        OUTPUTS:
        Updates the compiled_prompt_sets attribute with a dictionary where keys are prompt set names and values are tuples of (prompt_dict, weight).
        """

        #If no prompt dict list is provided, this function defaults to use the dense mask as the only prompt set.
        if prompt_dict_list is None:
            
            if self.dense_prompt_set is None:
                raise ValueError("No prompt dict list provided and no dense prompt set found. Please run load_dense_prompt() first.")
            
            self.compiled_prompt_sets = { "dense_mask": (self.dense_prompt_set, 1.0) }
            return

        if prompt_set_weights is None:
            if not (len(prompt_dict_list) == len(prompt_set_names)):
                raise ValueError(f"prompt_dict_list and prompt_set_names must have the same length. Got {len(prompt_dict_list)} and {len(prompt_set_names)}.")
        else:
            if not (len(prompt_dict_list) == len(prompt_set_names) == len(prompt_set_weights)):
                raise ValueError(f"prompt_dict_list, prompt_set_names, and prompt_set_weights ""must all have the same length. Got {len(prompt_dict_list)}, {len(prompt_set_names)}, and {len(prompt_set_weights)}.")

        #Use np.isclose to account for potential float summing being funkyyy
        if not np.isclose(sum(prompt_set_weights), 1.0):
            raise ValueError("Sum of prompt set weights must be 1.0.")

        #I realized too late that compile_prompt_sets inside segmentation_util has the same name LOL. my apologies for the confusion :].
        self.compiled_prompt_sets = compile_prompt_sets(prompt_dict_list, prompt_set_names, prompt_set_weights)
        return


    def remove_distant_slices(self, tolerance_frames=3):
        """
        Remove predicted segmentation slices that are more than tolerance_frames
        away from any slice containing dense mask pixels.

        INPUTS:
        tolerance_frames: Number of slices to keep on either side of the dense mask slices (default is 3).

        OUTPUTS:
        Updates the predicted_seg attribute to only include slices within the specified tolerance of the dense mask.

        """

        if not hasattr(self, "predicted_seg"):
            raise AttributeError(
                "No predicted segmentation found. Run run_segmentation() first."
            )

        if self.mask is None or self.mask.sum() == 0:
            raise ValueError("No valid dense mask found.")

        # Find z-slices where dense mask exists
        dense_slices = np.where(self.mask.astype(bool).any(axis=(1, 2)))[0]

        if len(dense_slices) == 0:
            raise ValueError("Dense mask contains no foreground slices.")

        z_min = max(0, dense_slices.min() - tolerance_frames)
        z_max = min(self.predicted_seg.shape[0] - 1, dense_slices.max() + tolerance_frames)

        # Create cleaned prediction
        cleaned_seg = np.zeros_like(self.predicted_seg)
        cleaned_seg[z_min:z_max + 1] = self.predicted_seg[z_min:z_max + 1]

        self.predicted_seg = cleaned_seg

        if self.verbose:
            print(
                f"Kept slices {z_min} to {z_max}. "
                f"Removed predicted segmentation outside dense mask ±{tolerance_frames} slices."
            )

        return self.predicted_seg


    def run_segmentation_sets(self, propagation_style="default", weighting_strategy="average", threshold=0.0,no_external_propagation=False,no_obj_score=-1024.0,no_obj_replacement=-10.0,mask_input_as_output_by_set=None,):
        """Function that calls MedSAM2 inference on all compiled prompt sets and returns the aggregated logits as predicted segmentation.
        
        INPUTS:
        propagation_style: Propagation style for SAM2 inference (default is "default").
        weighting_strategy: Strategy for combining logits from different prompt sets. Options are "average" or "custom" (default is "average").
        threshold: Threshold for converting logits to binary segmentation (default is 0.0).
        no_external_propagation: If True, limits propagation to slices where the initial dense mask is present (default is False).
        no_obj_score: Logit score indicating no object presence (default is -1024.0).
        no_obj_replacement: Replacement logit score for slices where no object is detected (default is -10.0).
        mask_input_as_output_by_set: Dictionary specifying whether to use mask input as output for each prompt set. If None, uses the predictor's default setting.

        OUTPUTS:
        predicted_seg: Final binary segmentation result after combining prompt set predictions.
        predicted_logits: Final logits after combining prompt set predictions.
        """


        if self.compiled_prompt_sets is None:
            raise ValueError("No compiled prompt sets found. Run compile_prompt_sets() first.")

        # Remember the predictor's original/default setting
        default_mask_input_as_output = (
            self.predictor.use_mask_input_as_output_without_sam
        )

        # An empty dictionary means no prompt-set-specific overrides
        if mask_input_as_output_by_set is None:
            mask_input_as_output_by_set = {}

        #if external propagation is turned off, propagation is limited to the slices where the initial dense mask is present. greatly optimizes segmentation
        if no_external_propagation:
            z = np.where(np.any(self.mask, axis=(1, 2)))[0]
            bounds = (z[0], z[-1])
        else:
            bounds = None

        logits_per_set = []
        weights_used = []

        self.segs_per_set = {}
        self.logits_per_set = {}

        #Loop through all prompt sets and run individual segmentations
        for num, (set_name, set_data) in enumerate(self.compiled_prompt_sets.items()):
            
            prompt_set, weight = set_data
            
            if weight is None or weight > 0.0:
                use_mask_as_output = mask_input_as_output_by_set.get(
                    set_name,
                    default_mask_input_as_output,
                )

                self.predictor.use_mask_input_as_output_without_sam = (
                    use_mask_as_output
                )

                print(f"Running segmentation for prompt set '{set_name}' with slices: {list(prompt_set.keys())}")

                pred_seg, pred_logits = run_medsam2_inference_from_arrays(
                    vol=self.img,
                    predictor=self.predictor,
                    image_size=512,#SET BACK TO 512 AFTER TESTING REGULAR SAM2
                    prompts_by_slice=prompt_set,
                    p_low=1.0,
                    p_high=99.0,
                    threshold=threshold,
                    propagation_style=propagation_style,
                    slice_bounds=bounds
                )

                pred_logits[np.isclose(pred_logits, no_obj_score)] = no_obj_replacement

                self.segs_per_set[set_name] = pred_seg
                self.logits_per_set[set_name] = pred_logits

                #Store logits for every prompt set for later fusion
                logits_per_set.append(pred_logits)

                if weight > 0.0:
                    weights_used.append(weight)

        #Basic, equal weighted logit averaging.
        if weighting_strategy == "average":
            final_logits = np.mean(logits_per_set, axis=0)

        elif weighting_strategy == "custom":
            #Creating empty frame to add all averaged logits to
            final_logits = np.zeros_like(logits_per_set[0], dtype=np.float32)

            for nr, weight_factor in enumerate(weights_used):
                final_logits += logits_per_set[nr] * weight_factor

        else:
            raise ValueError(f"Unknown weighting_strategy: {weighting_strategy}")

        self.predicted_seg = (final_logits > threshold).astype(np.uint8)
        self.predicted_logits = final_logits

        return self.predicted_seg, self.predicted_logits