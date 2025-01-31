from models.sled import apply_sled_to_volume
from utility import image_util as iu
import nibabel as nib
import yaml

def calculate_maps(sled, data_4d, mask_3d, config):
    fitted_signals_map, t2s_map, amps_map, sigma_map = apply_sled_to_volume(sled, data_4d)
    amps_map = iu.amps_sum2one(amps_map)
    mwf_map = iu.mwf_production(t2s_map, amps_map, config['mwf_cutoff'])
    residuals_map = fitted_signals_map - data_4d
    if config['mask_mwf_map']:
        mwf_map = mwf_map * mask_3d  # mask the mwf map
    results =  {
        'fitted_signals_map': fitted_signals_map,
        't2s_map': t2s_map,
        'amps_map': amps_map,
        'sigma_map': sigma_map,
        'mwf_map': mwf_map,
        'residuals_map': residuals_map,
        }
    return results
    
    
def save_results(results, affine, header, config):
    # save parameter maps to nifti files and dump the configs as a nifti extension (code=6 specifies a comment as a convention) 
    extension = nib.nifti1.Nifti1Extension(6, yaml.dump(config).encode())
    header.extensions.append(extension)
    header['descrip'] = config['io']['descrip']
    
    # iterate through requested map names and save if they exist in results
    for map_name in config['postprocessing']['save_maps']:
        dict_key = f"{map_name}_map"  # add _map suffix to match dictionary keys
        if dict_key in results:
            nib.save(
                nib.Nifti1Image(results[dict_key], affine, header),
                f"{config['io']['save_folder_path']}/{config['io']['save_prefix']}_{map_name}.nii.gz"
            )
