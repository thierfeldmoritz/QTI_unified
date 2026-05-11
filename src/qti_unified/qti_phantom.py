import os
from signal import signal
import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np
import scipy.io as sio
from matplotlib.widgets import Slider
from utils.dtd_math import (
    DTI_signal,
    QTI_signal,
    S0_hat,
    c_c,
    c_m,
    c_md,
    c_mu,
    compute_cumulant_tensors,
    convert_3x3_to_1x6,
    fa,
    k_bulk,
    k_mu,
    k_shear,
    md,
    mk,
    reorder_mfs,
    ufa,
    v_iso,
    v_md,
    v_shear,
    scaled_noise_from_snr,
    rician_complex,
)
from utils.dtd_utils import load_dtens_from_file, read_xps_mat


class QTI_Phantom:
    class RegionOfInterest:
        def __init__(
            self,
            x_range,
            y_range,
            z_range,
            dtens_filepath=None,
            roi_id="ROI",
            poi_id="POI",
            SNR=30,
            S0=1,
        ):
            self.roi_id = roi_id
            self.poi_id = poi_id
            self.x_range = x_range  # Tuple (x_start, x_end)
            self.y_range = y_range  # Tuple (y_start, y_end)
            self.z_range = z_range  # Tuple (z_start, z_end)

            self.dtens_filepath = dtens_filepath
            self.dtens = convert_3x3_to_1x6(load_dtens_from_file(dtens_filepath))
            self.dtens *= 1e9
            self.mfs = (
                None  # Model fitting structure that contains S0, dtens, ctens of dtd
            )
            self.dps = (
                None  # Diffusion parameter structure that contains QTI micro parameters
            )
            self.bulk_signal_value = None
            self.signal = None
            self.SNR = SNR
            self.S0 = S0
            self.signal_noisy = None

        def __repr__(self):
            return f"ROI {self.roi_id}) with POI {self.poi_id}."

    def __init__(
        self, Nx, Ny, Nz, dtens_filepath=None, btens_filepath=None, phantom_id="phantom", SNR=30, S0=1
    ):
        self.Nx = Nx
        self.Ny = Ny
        self.Nz = Nz
        self.dtens_filepath = dtens_filepath
        self.btens_filepath = btens_filepath
        self.phantom_id = phantom_id

        self.btens, self.dtens = self.load_tensors()
        self.mfs = None  # Model fitting structure that contains S0, dtens, ctens of dtd
        self.dps = (
            None  # Diffusion parameter structure that contains QTI micro parameters
        )

        # store the (1D) bulk signal value to broadcast
        self.bulk_signal_value = None
        # final phantom signal including ROIs
        self.signal = None
        self.SNR = SNR
        self.S0 = S0
        self.signal_noisy = None

        # List to hold 3D regions of interest (ROIs)
        self.rois = []
        self.pois = []

        self.mask = None
    '''
    # def init_tensors(self):
    #     dtens = convert_3x3_to_1x6(load_dtens_from_file(self.dtens_filepath))

    #     btens = read_xps_mat(self.btens_filepath)
    #     # btens.shape: (54, 6)

    #     # Broadcast and copy the dtens
    #     dtens_expanded = dtens[None, None, None, None, :, :]
    #     dtens_broadcast = np.broadcast_to(
    #         dtens_expanded,
    #         (self.Nx, self.Ny, self.Nz, btens.shape[0], dtens.shape[0], dtens.shape[1]),
    #     )
    #     dtens_final = np.copy(dtens_broadcast)

    #     # Broadcast and copy the btens
    #     btens_expanded = btens[None, None, None, :, None, :]
    #     btens_broadcast = np.broadcast_to(
    #         btens_expanded,
    #         (self.Nx, self.Ny, self.Nz, btens.shape[0], dtens.shape[0], btens.shape[1]),
    #     )
    #     # since we dont have to manipulate entries in btens, can probably skip the copy
    #     # btens_final = np.copy(btens_broadcast)
    #     btens_final = btens_broadcast

    #     # dtens_final.shape: (88, 88, 20, 54, 100, 6)
    #     # btens_final.shape: (88, 88, 20, 54, 100, 6)

    #     return btens_final, dtens_final
    '''    
    def load_tensors(self):
        """
        Load the bulk phantom tensors without broadcasting:
          - dtens: (n_dtens, 6)
          - btens: (n_btens, 6)
        """
        # Load and convert the bulk phantom dtens (resulting shape: (n_bulk, 6))
        bulk_dtens = convert_3x3_to_1x6(load_dtens_from_file(self.dtens_filepath))

        self.dtens = bulk_dtens[
            None, :, :
        ]   # shape: (1, n_dtens, 6)

        # Load the bulk b-tensors (expected shape: (n_btens, 6))
        bulk_btens = read_xps_mat(self.btens_filepath)

        self.btens = bulk_btens[
            :, None, :
        ]   # shape: (n_btens, 1, 6)

        return self.btens*1e-9, self.dtens*1e9

    def init_tensors(self):
        """
        Initialize the bulk phantom tensors:
          - dtens: (Nx, Ny, Nz, 54, n_bulk, 6)
          - btens: (Nx, Ny, Nz, 54, n_bulk, 6)
        """
        # Load and convert the bulk phantom dtens (resulting shape: (n_bulk, 6))
        bulk_dtens = convert_3x3_to_1x6(load_dtens_from_file(self.dtens_filepath))

        # Load the bulk b-tensors (expected shape: (54, 6))
        bulk_btens = read_xps_mat(self.btens_filepath)
        self.btens_orig = (
            bulk_btens  # keep the original for later ROI-specific broadcasting
        )

        # Broadcast the dtens: expand then broadcast to shape (Nx, Ny, Nz, 54, n_bulk, 6)
        dtens_expanded = bulk_dtens[
            None, None, None, None, :, :
        ]  # shape: (1,1,1,1,n_bulk,6)
        self.dtens = np.broadcast_to(
            dtens_expanded,
            (
                self.Nx,
                self.Ny,
                self.Nz,
                bulk_btens.shape[0],
                bulk_dtens.shape[0],
                bulk_dtens.shape[1],
            ),
        ).copy()

        # Broadcast the btens: expand then broadcast to shape (Nx, Ny, Nz, 54, n_bulk, 6)
        btens_expanded = bulk_btens[
            None, None, None, :, None, :
        ]  # shape: (1,1,1,54,1,6)
        self.btens = np.broadcast_to(
            btens_expanded,
            (
                self.Nx,
                self.Ny,
                self.Nz,
                bulk_btens.shape[0],
                bulk_dtens.shape[0],
                bulk_btens.shape[1],
            ),
        )
        return self.btens, self.dtens

    def roi_dtd_dim_check(self):
        # Iterate through each ROI, load its dtens, and print its dtd tensor count (dimension -2)
        roi_tensor_counts = []
        for roi in self.rois:
            dtens_roi = roi.dtens
            roi_dt_count = dtens_roi.shape[-2]
            roi_tensor_counts.append(roi_dt_count)
            print(
                f"ROI {roi.roi_id} with POI {roi.poi_id}, dtd tensor count: {roi_dt_count}"
            )

        phantom_dt_count = self.dtens.shape[-2]
        print(f"Phantom dtd tensor count: {phantom_dt_count}")

        # Ensure all ROIs have the same dtd tensor count.
        # if not all(count == roi_tensor_counts[0] for count in roi_tensor_counts):
        #     raise ValueError(
        #         "ROIs are incompatible, their dtd tensor count must match."
        #     )

        # roi_dt_count = roi_tensor_counts[0]

        # # If the phantom has fewer dtd tensors than the ROIs, replicate its dtens along the -2 axis
        # if phantom_dt_count < roi_dt_count:
        #     diff = roi_dt_count - phantom_dt_count
        #     # Copy the last tensor along the -2 axis
        #     last_entry = np.take(self.dtens, indices=[-1], axis=-2)
        #     # Replicate this tensor diff times along the -2 axis
        #     extra = np.repeat(last_entry, diff, axis=-2)
        #     self.dtens = np.concatenate([self.dtens, extra], axis=-2)
        #     print(
        #         f"Replicated phantom dtens along axis -2. New phantom dtd tensor count: {self.dtens.shape[-2]}. Experimental (not verified)."
        #     )

    def apply_rois(self):
        # DEPRACATED: executed in add_cuboid_roi now
        """
        Apply a region of interest to the phantom.

        Parameters:
        roi (RegionOfInterest): The region of interest to apply.
        Must be done AFTER initializing the phantom's tensors.
        Perhaps define property for num of meas. and max. num dtd entries for indexing, safer.
        """
        for roi in self.rois:
            dtens = convert_3x3_to_1x6(load_dtens_from_file(roi.dtens_filepath))

            dtens_expanded = dtens[None, None, None, None, :, :]

            target_shape = (
                roi.x_range[1] - roi.x_range[0],
                roi.y_range[1] - roi.y_range[0],
                roi.z_range[1] - roi.z_range[0],
                self.btens.shape[-3],
                self.dtens.shape[-2],
                self.dtens.shape[-1],
            )
            roi_dtens = np.broadcast_to(dtens_expanded, target_shape)

            # roi_dtens.shape: (20, 20, 15, 54, 100, 6)
            # NOTE: DTD in ROI, replace not add up?
            self.dtens[
                roi.x_range[0] : roi.x_range[1],
                roi.y_range[0] : roi.y_range[1],
                roi.z_range[0] : roi.z_range[1],
                :,
                :,
                :,
            ] = roi_dtens

            # self.dtens[
            #     roi.x_range[0] : roi.x_range[1],
            #     roi.y_range[0] : roi.y_range[1],
            #     roi.z_range[0] : roi.z_range[1],
            #     :,
            #     :,
            #     :,
            # ] += roi_dtens

    def generate_signal(self):
        # DEPRACATED: use generate_signal_flex instead, equivalence verified
        """
        Generate dMRI signal across the phantom.

        Returns:
        np.ndarray: The generated signal.
        """
        self.signal = DTI_signal(self.dtens, self.btens)

        # average over dtens/dtd dimension:
        self.signal = np.mean(self.signal, axis=-1)

        return self.signal

    def generate_cum_exp_signal(self):
        """
        Generate 2nd order cumulant expansion signal for the phantom and its ROIs.
        Requires that d and c-tensor have been computed. Purely 2nd order signal in b.
        """
        if self.mfs is None or any(roi.mfs is None for roi in self.rois):
            self.compute_phantom_cumulant_tensors()

        bulk_signal_value = QTI_signal(self.mfs, self.btens) # shape: (n_btens=54)
        
        # Broadcast the bulk signal value to the full phantom shape: (Nx, Ny, Nz, 54)
        signal_expanded = bulk_signal_value[
            None, None, None, :
        ]  # shape: (1,1,1,54)
        bulk_signal = np.broadcast_to(
            signal_expanded,
            (
                self.Nx,
                self.Ny,
                self.Nz,
                self.btens.shape[0],
            ),
        ).copy()

        if self.rois:
            for roi in self.rois:

                roi_signal = QTI_signal(roi.mfs, self.btens)

                roi_dim_x = roi.x_range[1] - roi.x_range[0]
                roi_dim_y = roi.y_range[1] - roi.y_range[0]
                roi_dim_z = roi.z_range[1] - roi.z_range[0]

                roi_signal_expanded = roi_signal[
                    None, None, None, :
                ]  # shape: (1,1,1,54)
                roi_signal_broadcast = np.broadcast_to(
                    roi_signal_expanded,
                    (
                        roi_dim_x,
                        roi_dim_y,
                        roi_dim_z,
                        self.btens.shape[0]
                    ),
                )
                
                bulk_signal[
                    roi.x_range[0] : roi.x_range[1],
                    roi.y_range[0] : roi.y_range[1],
                    roi.z_range[0] : roi.z_range[1],
                    :,
                ] = roi_signal_broadcast  # roi_signal

                roi.bulk_signal_value = roi_signal
                roi.signal = roi_signal_broadcast

        self.bulk_signal_value = bulk_signal_value
        self.signal = bulk_signal
        return self.signal

    def generate_signal_flex(self):
        """ 
        Generate the dMRI signal for the phantom in two steps:
          1. Compute the bulk phantom signal with the original tensors.
          2. For each ROI, load the ROI-specific dtens (which may have a different number of tensors),
             broadcast them and the (original) btens to the ROI volume, compute the ROI signal,
             and overwrite that region in the bulk signal.

        Returns:
            np.ndarray: The final phantom signal of shape (Nx, Ny, Nz, 54)
        """
        # --- Step 1: Calculate Bulk Signal ---
        # Compute the dMRI signal for the bulk phantom.
        # The DTI_signal function expects dtens and btens arrays with identical tensor dimensions.
        bulk_signal_value_temp = DTI_signal(self.dtens, self.btens) # shape: (n_btens=54, n_dtens)
        bulk_signal_value = np.mean(
            bulk_signal_value_temp, axis=-1
        )  # average over the tensor dimension (last axis) shape: (54)
        
        # Broadcast the bulk signal value to the full phantom shape: (Nx, Ny, Nz, 54)
        signal_expanded = bulk_signal_value[
            None, None, None, :
        ]  # shape: (1,1,1,54)
        bulk_signal = np.broadcast_to(
            signal_expanded,
            (
                self.Nx,
                self.Ny,
                self.Nz,
                self.btens.shape[0],
            ),
        ).copy()

        # Initialize a boolean mask for ROIs (False everywhere)
        self.mask = np.zeros((self.Nx, self.Ny, self.Nz), dtype=np.uint8)
        
        # --- Step 2: Apply ROI Signal Modifications ---
        for roi in self.rois:
            # Calculate the size of the ROI region.
            roi_dim_x = roi.x_range[1] - roi.x_range[0]
            roi_dim_y = roi.y_range[1] - roi.y_range[0]
            roi_dim_z = roi.z_range[1] - roi.z_range[0]

            # Load and convert the ROI-specific dtens.
            roi_dtens = roi.dtens
            # number of diffusion tensors in the ROI dtd
            # n_roi = roi_dtens.shape[0]

            roi_dtens_expanded = roi_dtens[
                None, :, :
            ]  # shape: (1, n_roi,6)

            # should match
            roi_btens_expanded = self.btens

            # roi_btens_expanded = self.btens_orig[
            #     :, None, :
            # ] # shape: (54,1,6)

            # Compute the ROI signal before fully broadbasting using the ROI-specific dtens and btens.
            roi_signal_temp = DTI_signal(roi_dtens_expanded, roi_btens_expanded) # shape: (54, n_roi)
            # average over dtd dimension
            roi_signal = np.mean(
                roi_signal_temp, axis=-1
            )  # shape: (54)

            # Broadcast THE SIGNAL to match the ROI spatial dimensions and the b-tensor dimension.
            # The desired shape is: (roi_dim_x, roi_dim_y, roi_dim_z, 54
            roi_signal_expanded = roi_signal[
                None, None, None, :
            ]  # shape: (1,1,1,54)
            roi_signal_broadcast = np.broadcast_to(
                roi_signal_expanded,
                (
                    roi_dim_x,
                    roi_dim_y,
                    roi_dim_z,
                    self.btens.shape[0]
                ),
            )
            '''
            # Broadcast ROI dtens to match the ROI spatial dimensions and the b-tensor dimension.
            # The desired shape is: (roi_dim_x, roi_dim_y, roi_dim_z, 54, n_roi, 6)
            roi_dtens_expanded = roi_dtens[
                None, None, None, None, :, :
            ]  # shape: (1,1,1,1, n_roi,6)
            roi_dtens_broadcast = np.broadcast_to(
                roi_dtens_expanded,
                (
                    roi_dim_x,
                    roi_dim_y,
                    roi_dim_z,
                    self.btens.shape[3],
                    n_roi,
                    roi_dtens.shape[1],
                ),
            )

            # Prepare an ROI-specific btens using the original bulk btens.
            # We want to broadcast btens_orig (shape: (54,6)) to: (roi_dim_x, roi_dim_y, roi_dim_z, 54, n_roi, 6)
            roi_btens_expanded = self.btens_orig[
                None, None, None, :, None, :
            ]  # shape: (1,1,1,54,1,6)
            roi_btens_broadcast = np.broadcast_to(
                roi_btens_expanded,
                (
                    roi_dim_x,
                    roi_dim_y,
                    roi_dim_z,
                    self.btens.shape[3],
                    n_roi,
                    self.btens_orig.shape[1],
                ),
            )

            # Compute the ROI signal using the ROI-specific dtens and btens.
            roi_signal_temp = DTI_signal(roi_dtens_broadcast, roi_btens_broadcast)
            roi_signal = np.mean(
                roi_signal_temp, axis=-1
            )  # shape: (roi_dim_x, roi_dim_y, roi_dim_z, 54)

            roi.signal = roi_signal
            '''            

            bulk_signal[
                roi.x_range[0] : roi.x_range[1],
                roi.y_range[0] : roi.y_range[1],
                roi.z_range[0] : roi.z_range[1],
                :,
            ] = roi_signal_broadcast  # roi_signal

            roi.bulk_signal_value = roi_signal
            roi.signal = roi_signal_broadcast

            # Set the mask to True for this ROI's voxels
            self.mask[
                roi.x_range[0] : roi.x_range[1],
                roi.y_range[0] : roi.y_range[1],
                roi.z_range[0] : roi.z_range[1],
            ] += 1
        
            self.bulk_signal_value = bulk_signal_value
            self.signal = bulk_signal
        
        return self.signal

    def compute_phantom_cumulant_tensors(self):
        # Load the diffusion tensors for the phantom
        dtens = convert_3x3_to_1x6(load_dtens_from_file(self.dtens_filepath))
        dtens *= 1e9
        # dtens = np.squeeze(self.dtens)
        # Compute the cumulant tensors for the bulk phantom
        d, c = compute_cumulant_tensors(dtens)
        self.mfs = np.concatenate([np.ones((1, 1)), d, c], axis=-1)
        # print("Computed cumulant tensors for the bulk phantom.")

        if not self.rois or len(self.rois) == 0:
            print("No ROIs defined. Skipping ROI cumulant tensor computation.")
            return
        for roi in self.rois:
            # Load the diffusion tensors for the ROI
            # why did we not make this an attribute of the ROI?
            dtens = roi.dtens
            # Compute the cumulant tensors for the ROI
            d, c = compute_cumulant_tensors(dtens)
            roi.mfs = np.concatenate([np.ones((1, 1)), d, c], axis=-1)
        # print("Computed cumulant tensors for ROIs.")
        pass

    def compute_qti_invariants(
        self,
        invariants=[
            S0_hat,
            md,
            v_md,
            v_shear,
            v_iso,
            c_md,
            c_mu,
            ufa,
            c_m,
            fa,
            c_c,
            mk,
            k_bulk,
            k_shear,
            k_mu,
        ],
    ):
        # make a dict of invars and their values for bulk and rois
        # invar names is input str list or so
        # compute invars with dtd math and store in dict dps
        if self.mfs is None:
            self.compute_phantom_cumulant_tensors()

        self.dps = {}
        for fn in invariants:
            # Call the function with self.mfs (adjust as needed)
            # have to reorder mfs due to differing voigt notations
            value = fn(reorder_mfs(self.mfs))
            # Use the function's __name__ as key to store the output
            self.dps[fn.__name__] = value
        # print("⭐ Computed scalar invariants for the bulk phantom.")

        if not self.rois or len(self.rois) == 0:
            print("No ROIs defined. Skipping ROI cumulant tensor computation.")
            return
        for roi in self.rois:
            if roi.mfs is None:
                roi.compute_phantom_cumulant_tensors()
            roi.dps = {}
            for fn in invariants:
                # Call the function with self.mfs (adjust as needed)
                value = fn(roi.mfs)
                # Use the function's __name__ as key to store the output
                roi.dps[fn.__name__] = value
        # print("⭐ Computed scalar invariants for ROIs.")
        pass

    def generate_parameter_map(self, invariant="ufa"):
        """
        Generate parameter maps (volumes) for the phantom and its ROIs.

        """
        if self.dps is None:
            self.compute_qti_invariants()

        if invariant not in self.dps:
            raise ValueError(
                f"Invariant '{invariant}' not found in computed parameters."
            )

        # Generate a numpy array with the dimensions of the phantom
        param_map = np.zeros((self.Nx, self.Ny, self.Nz))

        # Fill the array with the chosen parameter value
        param_value = self.dps[invariant]
        param_map.fill(param_value)

        if self.rois:
            for roi in self.rois:
                if invariant not in roi.dps:
                    raise ValueError(
                        f"Invariant '{invariant}' not found in ROI {roi.roi_id} parameters."
                    )

                x_start, x_end = roi.x_range
                y_start, y_end = roi.y_range
                z_start, z_end = roi.z_range

                param_map[x_start:x_end, y_start:y_end, z_start:z_end] = roi.dps[
                    invariant
                ]

        return param_map

    def add_cuboid_roi(
        self, x_range, y_range, z_range, dtens_filepath=None, roi_id="ROI", poi_id="POI", SNR=30, S0=1
    ):
        """
        Add a cuboid 3D region of interest to the phantom.

        Parameters:
           x_range (tuple): A tuple (x_start, x_end).
           y_range (tuple): A tuple (y_start, y_end).
           z_range (tuple): A tuple (z_start, z_end).
           properties (dict): A dictionary of properties shared by the ROI.

        Returns:
           RegionOfInterest: The created ROI instance.
        """
        roi = self.RegionOfInterest(
            x_range, y_range, z_range, dtens_filepath, roi_id, poi_id, SNR, S0
        )
        self.rois.append(roi)
        return roi

    def get_rois(self):
        """Return the list of defined regions of interest."""
        return self.rois
    
    def scale_regionwise_S0(self, bulk_s0=None):
        """
        Background (mask==0) uses bulk_snr/bulk_s0.
        Each ROI uses roi.snr / roi.s0.
        """
        if bulk_s0 is None:
            bulk_s0 = self.S0

        if self.signal is None:
            self.generate_signal_flex()

        S = self.signal
        out = S.copy()  # ensure writeable

        # --- background ---
        # S has shape (Nx, Ny, Nz, Nb)
        Nx, Ny, Nz, Nb = S.shape
        if self.mask.shape != (Nx, Ny, Nz):
            raise ValueError(f"mask shape {self.mask.shape} does not match signal shape {S.shape[:-1]}")
        
        mask_expanded = np.repeat(self.mask[..., None], Nb, axis=-1)  # shape (Nx, Ny, Nz, Nb)

        bg = (mask_expanded == 0)  # broadcast mask over last (btens) axis
        if bg.any():
            out_bg = S*bulk_s0
            out[bg] = out_bg[bg]

        # --- ROIs ---
        for roi in self.rois:
            xs = slice(roi.x_range[0], roi.x_range[1])
            ys = slice(roi.y_range[0], roi.y_range[1])
            zs = slice(roi.z_range[0], roi.z_range[1])

            sub = S[xs, ys, zs, :]
            out[xs, ys, zs, :] = sub*roi.S0
            roi.signal = out[xs, ys, zs, :]  # optional: keep per-ROI result

        self.signal = out
        return out

    def add_regionwise_rician_noise(self, bulk_snr=None, bulk_s0=None, rng=None):
        """
        Background (mask==0) uses bulk_snr/bulk_s0.
        Each ROI uses roi.snr / roi.s0.
        """
        if bulk_snr is None:
            bulk_snr = self.SNR
        if bulk_s0 is None:
            bulk_s0 = self.S0

        if self.signal is None:
            self.generate_signal_flex()
        rng = np.random.default_rng(rng) if not isinstance(rng, np.random.Generator) else rng

        S = self.signal
        out = S.copy()  # ensure writeable

        # --- background ---
        # S has shape (Nx, Ny, Nz, Nb)
        Nx, Ny, Nz, Nb = S.shape
        if self.mask.shape != (Nx, Ny, Nz):
            raise ValueError(f"mask shape {self.mask.shape} does not match signal shape {S.shape[:-1]}")
        
        mask_expanded = np.repeat(self.mask[..., None], Nb, axis=-1)  # shape (Nx, Ny, Nz, Nb)

        bg = (mask_expanded == 0)  # broadcast mask over last (btens) axis
        if bg.any():
            out_bg = scaled_noise_from_snr(S, bulk_snr, bulk_s0, rng)
            out[bg] = out_bg[bg]

        # --- ROIs ---
        for roi in self.rois:
            xs = slice(roi.x_range[0], roi.x_range[1])
            ys = slice(roi.y_range[0], roi.y_range[1])
            zs = slice(roi.z_range[0], roi.z_range[1])

            sub = S[xs, ys, zs, :]
            out[xs, ys, zs, :] = scaled_noise_from_snr(sub, roi.SNR, roi.S0, rng)
            roi.noisy_signal = out[xs, ys, zs, :]  # optional: keep per-ROI result

        self.signal_noisy = out
        return out

    def add_regionwise_rician_noise_stack(self, n_realizations, bulk_snr=None, bulk_s0=None,
                                        seed=None, append=False):
        """
        Generate n_realizations of regionwise Rician noise and stack along z.
        Deterministic via SeedSequence.spawn: same seed + same n_realizations
        => identical results. Each realization uses an independent child RNG.

        Output shape:
            (Nx, Ny, R, Nb)  if fresh (Nz==1 is required)
            (Nx, Ny, R_old+R, Nb) if append=True and compatible.
        """
        import numpy as np

        if bulk_snr is None: bulk_snr = self.SNR
        if bulk_s0  is None: bulk_s0  = self.S0

        if self.signal is None:
            self.generate_signal_flex()

        S = self.signal  # (Nx, Ny, Nz, Nb), expecting Nz==1
        Nx, Ny, Nz, Nb = S.shape
        if Nz != 1:
            raise ValueError(f"Expected Nz==1 for 2D phantom; got Nz={Nz}")
        if self.mask.shape != (Nx, Ny, Nz):
            raise ValueError(f"mask shape {self.mask.shape} does not match signal shape {S.shape[:-1]}")

        # Independent RNG per realization (deterministic w.r.t seed & n_realizations)
        ss = np.random.SeedSequence(seed)
        rngs = [np.random.default_rng(s) for s in ss.spawn(n_realizations)]

        # Precompute helpers
        mask_expanded = np.repeat(self.mask[..., None], Nb, axis=-1)  # (Nx,Ny,1,Nb)
        stack = np.empty((Nx, Ny, n_realizations, Nb), dtype=S.dtype)

        for r, rng in enumerate(rngs):
            out_r = S.copy()

            # Background
            bg = (mask_expanded == 0)
            if bg.any():
                out_bg = scaled_noise_from_snr(S, bulk_snr, bulk_s0, rng)
                out_r[bg] = out_bg[bg]

            # ROIs
            for roi in self.rois:
                xs = slice(roi.x_range[0], roi.x_range[1])
                ys = slice(roi.y_range[0], roi.y_range[1])
                z0 = slice(0, 1)
                sub = S[xs, ys, z0, :]
                out_r[xs, ys, z0, :] = scaled_noise_from_snr(sub, bulk_snr, roi.S0, rng) # changed ROI SNR to bulk_snr, why would that differ?

            # Place realization as its own z-slice
            stack[:, :, r:r+1, :] = out_r

        if append and (self.signal_noisy is not None):
            if self.signal_noisy.shape[:2] != (Nx, Ny) or self.signal_noisy.shape[3] != Nb:
                raise ValueError(f"Incompatible existing signal_noisy shape {self.signal_noisy.shape}")
            self.signal_noisy = np.concatenate([self.signal_noisy, stack], axis=2)
        else:
            self.signal_noisy = stack

        # Optional provenance
        self.noise_provenance = dict(seed=seed, n_realizations=n_realizations, scheme='spawn_per_realization')
        return self.signal_noisy

    def regionwise_rician_noise_from_snr_stack(self, n_realizations, bulk_snr=None, bulk_s0=None,
                                        seed=None, append=False):
        """
        Region-wise Rician noise (complex construction) stacked along z:
        output shape: (Nx, Ny, R, Nb)   with R = n_realizations  (requires Nz==1).
        Background uses bulk_snr/bulk_s0; each ROI uses roi.SNR/roi.S0.
        Deterministic: same seed + same n_realizations -> identical stack.
        """
        if bulk_snr is None: bulk_snr = self.SNR
        if bulk_s0  is None: bulk_s0  = self.S0

        if self.signal is None:
            self.generate_signal_flex()

        S = self.signal                      # (Nx, Ny, Nz, Nb), expect Nz==1
        Nx, Ny, Nz, Nb = S.shape
        if Nz != 1:
            raise ValueError(f"Expected Nz==1; got Nz={Nz}")
        if self.mask.shape != (Nx, Ny, Nz):
            raise ValueError(f"mask shape {self.mask.shape} != {S.shape[:-1]}")

        # independent, reproducible RNG per realization
        ss = np.random.SeedSequence(seed)
        rngs = [np.random.default_rng(s) for s in ss.spawn(n_realizations)]

        # expand mask only along last axis
        mask_expanded = np.repeat(self.mask[..., None], Nb, axis=-1)  # (Nx,Ny,1,Nb)
        stack = np.empty((Nx, Ny, n_realizations, Nb), dtype=S.dtype)

        for r, rng in enumerate(rngs):
            out_r = S.copy()

            # ---- background (mask==0) ----
            bg = (mask_expanded == 0)
            if bg.any():
                out_bg = rician_complex(S, bulk_snr, bulk_s0, rng)
                out_r[bg] = out_bg[bg]

            # ---- ROIs ----
            for roi in self.rois:
                xs = slice(roi.x_range[0], roi.x_range[1])
                ys = slice(roi.y_range[0], roi.y_range[1])
                z0 = slice(0, 1)  # Nz==1
                sub = S[xs, ys, z0, :]
                # use ROI-specific SNR/S0 (not bulk)
                out_r[xs, ys, z0, :] = rician_complex(sub, bulk_snr, roi.S0, rng) # changed ROI SNR to bulk_snr, why would that differ?

            # place this realization as its own z-slice in the stack
            stack[:, :, r:r+1, :] = out_r

        if append and (getattr(self, "signal_noisy", None) is not None):
            if self.signal_noisy.shape[:2] != (Nx, Ny) or self.signal_noisy.shape[3] != Nb:
                raise ValueError(f"Incompatible existing signal_noisy shape {self.signal_noisy.shape}")
            self.signal_noisy = np.concatenate([self.signal_noisy, stack], axis=2)
        else:
            self.signal_noisy = stack

        self.noise_provenance = dict(seed=seed, n_realizations=n_realizations,
                                    scheme="complex_rician_spawn_per_realization")
        return self.signal_noisy
    
    def signal_dropout(self, start_ind=0, end_ind=None, dropout_fraction=0.1, atten_factor=0.05,
                    roi_ids=None, seed=None, use_raw_signal=False):
        """
        Randomly attenuate a fraction of encodings in [start_ind, end_ind) by atten_factor,
        applied only within selected ROIs (roi_ids). Defaults to all ROIs.
        """
        if use_raw_signal:
            if self.signal is None:
                self.generate_signal_flex()
            S = self.signal  # (Nx, Ny, Nz, Nb)
        else:
            if self.signal_noisy is None:
                raise RuntimeError("No noisy signal available. Please generate noisy signal first or set use_raw_signal=True.")
            else:
                S = self.signal_noisy  # (Nx, Ny, Nz, Nb)    

        if S is None or S.ndim != 4:
            raise ValueError("self.signal must be a 4D array (Nx, Ny, Nz, Nb).")
        Nx, Ny, Nz, Nb = S.shape
        if Nb < 2:
            raise ValueError(f"Insufficient measurements in encoding dimension (Nb={Nb}).")

        if end_ind is None:
            end_ind = Nb
        if not (0 <= start_ind < end_ind <= Nb):
            raise ValueError(f"Invalid start_ind {start_ind} and end_ind {end_ind} for Nb={Nb}")
        if not (0.0 <= dropout_fraction <= 1.0):
            raise ValueError(f"dropout_fraction must be in [0,1], got {dropout_fraction}")
        if not (0.0 <= atten_factor <= 1.0):
            raise ValueError(f"atten_factor must be in [0,1], got {atten_factor}")

        span = end_ind - start_ind
        rng = np.random.default_rng(seed)

        # Build encoding mask
        if span == 0 or dropout_fraction == 0.0 or atten_factor == 1.0:
            emask = np.zeros(Nb, dtype=bool)
            self._dropout_mask = emask
            return
        n_drop = int(np.floor(dropout_fraction * span))
        if n_drop < 1:
            emask = np.zeros(Nb, dtype=bool)
            self._dropout_mask = emask
            return
        emask = np.zeros(Nb, dtype=bool)
        emask[start_ind + rng.choice(span, size=n_drop, replace=False)] = True

        # Select ROIs (default: all)
        if roi_ids is None:
            selected = list(self.rois)
        else:
            want = set(roi_ids)
            selected = [r for r in self.rois if r.roi_id in want]
            missing = want - {r.roi_id for r in selected}
            if missing:
                print(f"[signal_dropout] Warning: ROI ids not found: {sorted(missing)}")

        # Apply attenuation per ROI
        for roi in selected:
            xs, xe = roi.x_range
            ys, ye = roi.y_range
            zs, ze = roi.z_range

            S[xs:xe, ys:ye, zs:ze, emask] *= atten_factor

        if use_raw_signal:
            self.signal = S  # (Nx, Ny, Nz, Nb)
        else:
            self.signal_noisy = S  # (Nx, Ny, Nz, Nb)

        # Bookkeeping (global encoding mask used)
        self._dropout_mask = emask

        
    def load_btens_from_mat(self, filepath):
        """
        DEPR?
        Load the b-tensor from a MAT file.

        Parameters:
        filepath (str): Path to the MAT file.
        """
        pass

    def load_dtens_from_file(self, filepath):
        """
        DEPR?
        Load the diffusion tensors from a JSON file.

        Parameters:
        filepath (str): Path to the JSON file.
        """
        self.dtens = load_dtens_from_file(filepath)

    def save_signal_to_file(self, filepath="SynQTI_signal.nii", save_raw_signal=False):
        """
        Save the generated signal to a file.
        Parameters:
        filepath (str): Path to the file.
        """
        # Create a default affine transformation (identity)
        affine = np.eye(4)
        # Create a Nifti1 image from the signal (optionally use noisy signal)
        if self.signal_noisy is not None and not save_raw_signal:
            data_to_save = self.signal_noisy
        else:
            data_to_save = self.signal
        nifti_img = nib.Nifti1Image(data_to_save, affine)
        # Save the image to the specified file path
        nib.save(nifti_img, filepath)

    def save_dps_to_mat(self, filepath="QTI_dps.mat", swap_field_names = True):
        """
        Save the diffusion parameter structure (dps) to a MAT file.

        Parameters:
        filepath (str): Path to the MAT file.
        """
        if self.dps is None:
            self.compute_qti_invariants()

        dps_dict = {}
        for invariant, value in self.dps.items():
            # Generate parameter map for each invariant
            param_map = self.generate_parameter_map(invariant)
            dps_dict[invariant] = param_map
        # vs more minimal:
        # dps_dict = {inv: self.generate_parameter_map(inv) for inv in self.dps}

        if swap_field_names:
            dps_dict = _normalize_dps_fieldnames(dps_dict)

        # Save the dps_dict to a MAT file
        sio.savemat(filepath, {"dps": dps_dict})
        print(f"Diffusion parameter structure saved to {filepath}")

    '''
    DEPR no longer class method
    def plot_signal(
        self,
        slice_ind=None,
        meas_ind=0,
        cmap="gray",
        vmax=None,
        vmin=None,
        save=False,
        out_dir=None,
        skip_plot=False,
        plot_raw_signal=False,
    ):
        """
        Plot the generated signal with sliders to navigate through slices (z-dimension) and measurements (last dimension).

        Assumes self.signal is a 4D numpy array with shape (Nx, Ny, Nz, M), where M is the number of measurements.
        """
        if self.signal_noisy is not None and not plot_raw_signal:
            signal = self.signal_noisy
        elif self.signal is None:
            self.generate_signal()
            signal = self.signal
        else:
            signal = self.signal

        # Determine dimensions of the signal
        num_slices = signal.shape[2]
        num_meas = signal.shape[3]

        # Set default slice if not provided
        if slice_ind is None:
            slice_ind = num_slices // 2

        # Create the figure and axis for the image
        fig, ax = plt.subplots(figsize=(8, 6))
        plt.subplots_adjust(bottom=0.25)  # leave space for sliders

        # Display the initial image: slice along z and measurement index
        img = ax.imshow(
            signal[:, :, slice_ind, meas_ind], cmap=cmap, vmax=vmax, vmin=vmin
        )
        ax.set_title(f"Signal Slice {slice_ind}, Measurement {meas_ind}")
        fig.colorbar(img, ax=ax)

        # Create slider axes: one for slice and one for measurement
        ax_slice = plt.axes([0.15, 0.1, 0.65, 0.03], facecolor="lightgoldenrodyellow")
        ax_meas = plt.axes([0.15, 0.05, 0.65, 0.03], facecolor="lightgoldenrodyellow")

        # Create sliders with integer steps
        slider_slice = Slider(
            ax_slice,
            "Slice",
            0,
            num_slices - 1,
            valinit=slice_ind,
            valstep=1,
            valfmt="%d",
        )
        slider_meas = Slider(
            ax_meas,
            "Measurement",
            0,
            num_meas - 1,
            valinit=meas_ind,
            valstep=1,
            valfmt="%d",
        )

        # Update function to refresh image when sliders are changed
        def update(val):
            cur_slice = int(slider_slice.val)
            cur_meas = int(slider_meas.val)
            img.set_data(signal[:, :, cur_slice, cur_meas])
            ax.set_title(f"Signal Slice {cur_slice}, Measurement {cur_meas}")
            fig.canvas.draw_idle()

        slider_slice.on_changed(update)
        slider_meas.on_changed(update)

        # If save is True and an output directory is provided, save the current image
        if save and out_dir is not None:
            if not os.path.exists(out_dir):
                os.makedirs(out_dir)
            save_path = os.path.join(
                out_dir, f"signal_slice{slice_ind}_meas{meas_ind}.pdf"
            )
            plt.savefig(save_path, format="pdf", dpi=300, bbox_inches="tight")
            print(f"Figure saved to {save_path}")

        if skip_plot:
            plt.close(fig)
            return fig, ax, slider_slice, slider_meas

        plt.show()
        return fig, ax, slider_slice, slider_meas
    '''

    def plot_param_map(
        self,
        invariant="ufa",
        slice_ind=None,
        cmap="gray",
        vmax=None,
        vmin=None,
        save=False,
        out_dir=None,
        skip_plot=False,
    ):
        """
        Plot the generated parameter map with sliders to navigate through slices (z-dimension).

        Parameters:
        invariant (str): The key for the parameter map to plot.
        slice_ind (int): The initial slice index to display. Defaults to the middle slice.
        cmap (str): Colormap to use for the plot. Defaults to 'viridis'.
        vmax (float): Maximum value for the color scale. Defaults to None.
        vmin (float): Minimum value for the color scale. Defaults to None.
        save (bool): Whether to save the plot as a file. Defaults to False.
        out_dir (str): Directory to save the plot if save is True. Defaults to None.
        skip_plot (bool): If True, skip showing the plot. Defaults to False.
        """
        # Generate the parameter map
        param_map = self.generate_parameter_map(invariant)

        # # Replace NaN values with zero for plotting
        # param_map = np.nan_to_num(param_map)

        # Determine dimensions of the parameter map
        num_slices = param_map.shape[2]

        # Set default slice if not provided
        if slice_ind is None:
            slice_ind = num_slices // 2

        # Create the figure and axis for the image
        fig, ax = plt.subplots(figsize=(8, 6))
        plt.subplots_adjust(bottom=0.25)  # leave space for sliders

        # Display the initial image: slice along z
        img = ax.imshow(param_map[:, :, slice_ind], cmap=cmap, vmax=vmax, vmin=vmin)
        ax.set_title(f"Parameter Map Slice {slice_ind} ({invariant})")
        fig.colorbar(img, ax=ax)

        # Create slider axes for slice navigation
        ax_slice = plt.axes([0.15, 0.1, 0.65, 0.03], facecolor="lightgoldenrodyellow")

        # Create slider with integer steps
        slider_slice = Slider(
            ax_slice,
            "Slice",
            0,
            num_slices - 1,
            valinit=slice_ind,
            valstep=1,
            valfmt="%d",
        )

        # Update function to refresh image when slider is changed
        def update(val):
            cur_slice = int(slider_slice.val)
            img.set_data(param_map[:, :, cur_slice])
            ax.set_title(f"Parameter Map Slice {cur_slice} ({invariant})")
            fig.canvas.draw_idle()

        slider_slice.on_changed(update)

        # If save is True and an output directory is provided, save the current image
        if save and out_dir is not None:
            if not os.path.exists(out_dir):
                os.makedirs(out_dir)
            save_path = os.path.join(
                out_dir, f"param_map_{invariant}_slice{slice_ind}.pdf"
            )
            plt.savefig(save_path, format="pdf", dpi=300, bbox_inches="tight")
            print(f"Figure saved to {save_path}")

        if skip_plot:
            plt.close(fig)
            return fig, ax, slider_slice

        plt.show()
        return fig, ax, slider_slice

    def save_param_map_to_nii(self, invariant="ufa", filepath="QTI_param.nii"):
        """
        Save the generated parameter to a Nifti file.
        Parameters:
        filepath (str): Path to the file.
        """
        param_map = self.generate_parameter_map(invariant)

        # Create a default affine transformation (identity)
        affine = np.eye(4)
        # Create a Nifti1 image from the parameter
        nifti_img = nib.Nifti1Image(param_map, affine)
        # Save the image to the specified file path
        nib.save(nifti_img, filepath)

def plot_signal(
    signal,
    slice_ind=None,
    meas_ind=0,
    cmap="gray",
    vmax=None,
    vmin=None,
    save=False,
    out_dir=None,
    skip_plot=False,
    plot_raw_signal=False,
    ):
    """
    Plot the generated signal with sliders to navigate through slices (z-dimension) and measurements (last dimension).

    Assumes self.signal is a 4D numpy array with shape (Nx, Ny, Nz, M), where M is the number of measurements.
    """
    # Sanity check: expect a 4D numpy array
    if not isinstance(signal, np.ndarray) or signal.ndim != 4:
        raise ValueError(
            f"Expected 'signal' to be a 4D numpy array, got type {type(signal)} with shape {getattr(signal, 'shape', None)}"
        )

    # Determine dimensions of the signal
    num_slices = signal.shape[2]
    num_meas = signal.shape[3]

    # Set default slice if not provided
    if slice_ind is None:
        slice_ind = num_slices // 2

    # Create the figure and axis for the image

    fig, ax = plt.subplots(figsize=(8, 6))
    plt.subplots_adjust(bottom=0.25)  # leave space for sliders

    # Display the initial image: slice along z and measurement index
    img = ax.imshow(
        signal[:, :, slice_ind, meas_ind], cmap=cmap, vmax=vmax, vmin=vmin
    )
    ax.set_title(f"Signal Slice {slice_ind}, Measurement {meas_ind}")
    fig.colorbar(img, ax=ax)

    # Create slider axes: one for slice and one for measurement
    ax_slice = plt.axes([0.15, 0.1, 0.65, 0.03], facecolor="lightgoldenrodyellow")
    ax_meas = plt.axes([0.15, 0.05, 0.65, 0.03], facecolor="lightgoldenrodyellow")

    # Create sliders with integer steps
    slider_slice = Slider(
        ax_slice,
        "Slice",
        0,
        num_slices - 1,
        valinit=slice_ind,
        valstep=1,
        valfmt="%d",
    )
    slider_meas = Slider(
        ax_meas,
        "Measurement",
        0,
        num_meas - 1,
        valinit=meas_ind,
        valstep=1,
        valfmt="%d",
    )

    # Update function to refresh image when sliders are changed
    def update(val):
        cur_slice = int(slider_slice.val)
        cur_meas = int(slider_meas.val)
        img.set_data(signal[:, :, cur_slice, cur_meas])
        ax.set_title(f"Signal Slice {cur_slice}, Measurement {cur_meas}")
        fig.canvas.draw_idle()

    slider_slice.on_changed(update)
    slider_meas.on_changed(update)

    # If save is True and an output directory is provided, save the current image
    if save and out_dir is not None:
        if not os.path.exists(out_dir):
            os.makedirs(out_dir)
        save_path = os.path.join(
            out_dir, f"signal_slice{slice_ind}_meas{meas_ind}.pdf"
        )
        plt.savefig(save_path, format="pdf", dpi=300, bbox_inches="tight")
        print(f"Figure saved to {save_path}")

    if skip_plot:
        plt.close(fig)
        return fig, ax, slider_slice, slider_meas

    plt.show()
    return fig, ax, slider_slice, slider_meas

def plot_signal_per_voxel(signal, x, y, z, b_min=0, b_max=None, plot_raw_signal=False, fig=None, ax=None):
    """
    Plot the generated signal per voxel.
    """
    # Sanity check: expect a 4D numpy array
    if not isinstance(signal, np.ndarray) or signal.ndim != 4:
        raise ValueError(
            f"Expected 'signal' to be a 4D numpy array, got type {type(signal)} with shape {getattr(signal, 'shape', None)}"
        )

    if b_max is None:
        b_max = np.shape(signal)[-1]

    voxel_signal = signal[x, y, z, b_min:b_max]
    if fig is None or ax is None:
        fig, ax = plt.subplots()
    ax.plot(np.arange(voxel_signal.size), voxel_signal, marker="o")
    ax.set_xlabel("Measurement index")
    ax.set_ylabel("Signal intensity")
    ax.set_title(f"Signal at voxel ({x}, {y}, {z})")
    plt.tight_layout()
    plt.show()
    return fig, ax




    

def _normalize_dps_fieldnames(dps_dict):
    """
    Return a new dict with keys renamed to the target (MATLAB) casing.
    Unknown keys pass through unchanged. Collisions overwrite (last wins).
    """
    mapping = {
        # signal / misc
        's0': 's0', 's0_hat': 's0', 'nii_h': 'nii_h',

        # scalars
        'fa': 'FA', 'md': 'MD', 'ad': 'AD', 'rd': 'RD',
        'mk': 'MK', 'mkt': 'MKt', 'mki': 'MKi', 'mka': 'MKa',
        'op': 'OP', 'op2': 'OP2',

        # vectors / dirs / colors
        'ax_dir': 'ax_dir', 'fa_col': 'FA_col',

        # variances
        'v_md': 'V_MD', 'v_iso': 'V_iso', 'v_shear': 'V_shear',

        # covariances / tensors
        'c_md': 'C_MD', 'c_mu': 'C_mu', 'c_m': 'C_M', 'c_c': 'C_c',

        # micro-fraction
        'ufa': 'uFA',

        # moduli
        'k_bulk': 'K_bulk', 'k_shear': 'K_shear', 'k_mu': 'K_mu',
    }

    out = {}
    for k, v in dps_dict.items():
        # try exact, then case-insensitive
        tgt = mapping.get(k, mapping.get(k.lower(), k))
        out[tgt] = v
    return out