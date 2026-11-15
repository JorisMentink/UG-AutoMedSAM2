from pathlib import Path
import SimpleITK as sitk
from segmentation_util import compute_contour_average
import numpy as np
from scipy.ndimage import distance_transform_edt

class DataLoader():
    def __init__(self,parentfolder,subject_nr=0,volume_of_interest="CTVT",verbose=False):
        """
        Initializes the dataloader, based on LUNDPROBE dataset formatting.
        
        INPUTS:
        parentfolder: Path to the folder containing all subjects. Each subject should be in a separate folder.
        subject_nr: Index of the subject to load (default is 0).
        volume_of_interest: The volume of interest to load. Can be "CTVT" or "rectum" in original LUNDPROBE dataset.
        Verbose: If True, prints information about the loaded data (default is False).

        OUTPUTS:
        Generates dataloader attributes containing image, masks, uncertainty maps etc.
        """


        self.parentfolder = Path(parentfolder)
        self.subject_nr = subject_nr
        self.volume_of_interest = volume_of_interest
        self.verbose = verbose

        #Method only supports prostate volume (CTVT) and rectum as of now
        if self.volume_of_interest not in ["CTVT","rectum","bladder"]:
            raise ValueError("Volume of interest must be either 'CTVT', 'rectum' or 'bladder'")

        #Sorts subjects and selects subject folder based on subject number. Follows out-of-the-box LUNDPROBE formatting.
        self.subjects = sorted([p.name for p in self.parentfolder.iterdir() if p.is_dir()])
        self.subjectfolder = self.parentfolder / str(self.subjects[subject_nr]) / "MR_StorT2"
        
        self.subject_name = str(self.subjects[subject_nr])

        #Load paths for image, mask and uncertainty map -following out-of-the-box LUNDPROBE formatting
        img_path = self.subjectfolder / "image.nii.gz"


        #In case of other ROIs, mask uncertainty map and ground truth paths should be set as follows.
        if self.volume_of_interest == "CTVT":
            mask_path = self.subjectfolder / "nnUNetOutput/mask_CTVT_427_nnUNet.nii.gz"
            unc_path = self.subjectfolder / "nnUNetOutput/mask_CTVT_427_nnUNet_uncertaintyMap.nii.gz"
            gt_path = self.subjectfolder / "mask_CTVT_427.nii.gz"
                
        elif self.volume_of_interest == "rectum":
            mask_path = self.subjectfolder / "nnUNetOutput/mask_Rectum_nnUNet.nii.gz"
            unc_path = self.subjectfolder / "nnUNetOutput/mask_Rectum_nnUNet_uncertaintyMap.nii.gz"
            gt_path = self.subjectfolder / "mask_Rectum.nii.gz"
        
        #Load data as arrays
        self.img = sitk.GetArrayFromImage(sitk.ReadImage(str(img_path)))
        self.mask = sitk.GetArrayFromImage(sitk.ReadImage(str(mask_path))) > 0
        self.unc_map = sitk.GetArrayFromImage(sitk.ReadImage(str(unc_path)))
        self.gt = sitk.GetArrayFromImage(sitk.ReadImage(str(gt_path))) > 0
            
        #Load image spacing
        img_itk = sitk.ReadImage(str(img_path))
        spacing_sitk = img_itk.GetSpacing()  # (x, y, z)
        self.img_spacing = spacing_sitk[::-1]  # (z, y, x) #Necessary for inferior-to-superior processing in 3D MRI volumes

        if self.verbose:
            print(f"Loaded subject {self.subjects[subject_nr]} with volume of interest '{self.volume_of_interest}'")
            print(f"Image shape: {self.img.shape}, Mask shape: {self.mask.shape}, Uncertainty map shape: {self.unc_map.shape}, Ground truth shape: {self.gt.shape}")
            print(f"Image spacing (z, y, x): {self.img_spacing}")   

    #Loads observer recontours from LUNDPROBE dataset. Only run if applicable.
    def load_recontours(self):
        """
        Function that loads the observer recontours present in the data, expects the LUNDPROBE dataset format.

        INPUTS:
        None

        OUTPUTS:
        observer_recontours: List of numpy arrays containing the observer recontours.
        additionally adds observer_names attribute to the dataloader, containing the names of the observers.
        """
        
        if self.volume_of_interest == "CTVT":
            observer_paths = sorted((self.subjectfolder / "observerData").glob("mask_CTVT_427_step2_obs*.nii.gz"))

        elif self.volume_of_interest == "rectum":
            observer_paths = sorted((self.subjectfolder / "observerData").glob("mask_Rectum_step2_obs*.nii.gz"))

        self.observer_recontours = [
            sitk.GetArrayFromImage(sitk.ReadImage(str(p))) > 0
            for p in observer_paths
        ]

        self.observer_names = [p.name.split("obs")[-1].replace(".nii.gz", "") for p in observer_paths]

        return self.observer_recontours

    #Generates consensus recontour from observer recontours. Indices denotes what observers are included.
    def load_consensus(self,indices = [0,2,3]):
        """"
        Computes consensus of recontours for evaluation.

        INPUTS:
        indices: List of indices of observers to include in the consensus (default is [0, 2, 3]).

        OUTPUTS:
        consensus_recontour: Numpy array containing the consensus recontour.
        """

        if not hasattr(self, 'observer_recontours'):
            print("Load_recontours() was not executed yet. Executing now...")
            self.load_recontours()

        selected_recontours = [self.observer_recontours[i] for i in indices]
        self.consensus_recontour = compute_contour_average(selected_recontours, self.img_spacing) #Imported from segmentation_util
        
        return self.consensus_recontour

    #Generates summed recontour from observer recontours. Indices denotes what observers are included.
    def load_summed_recontours(self,indices = [0,2,3]):
        """"
        Computes summed recontours for evaluation.

        INPUTS:
        indices: List of indices of observers to include in the summation (default is [0, 2, 3]).

        OUTPUTS:
        summed_recontour: Numpy array containing the summed recontour.
        """

        if not hasattr(self, 'observer_recontours'):
            print("Load_recontours() was not executed yet. Executing now...")
            self.load_recontours()

        selected_recontours = [self.observer_recontours[i] for i in indices]
        self.summed_recontour = sum(selected_recontours)
        
        return self.summed_recontour
        

