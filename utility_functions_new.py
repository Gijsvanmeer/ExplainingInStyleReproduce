import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
import tqdm
from utility_functions_stylex_notebook import sindex_to_layer_idx_and_index

device = 'cuda' if torch.cuda.is_available() else 'cpu'

def load_model_state_dicts(model_path, stylex):
    """loads the state dict for a model path, 
    and if it is not a stylex model converts the keys to ones compatible with our stylex implementation"""
    model_dict = torch.load(model_path)

    if not stylex:
        tmp = torch.load(model_path)
        for key in tmp['g_ema'].keys():
          if key[:5] ==  'style':
            model_dict['g_ema']['mapping'+key[5:]]= model_dict['g_ema'][key]
            del model_dict['g_ema'][key]
          if key[-17:]=='modulation.weight':
            if 'rgb' in key:
              if key[6] == '1':
                model_dict['g_ema']['style_affines_rgb.0.modulation.weight']= model_dict['g_ema'][key]
                del model_dict['g_ema'][key]
              else:
                model_dict['g_ema']['style_affines_rgb.'+str(int(key[8:10].strip('.'))+1)+'.modulation.weight']= model_dict['g_ema'][key]
                del model_dict['g_ema'][key]
            else:
              if key[4] == '1':
                model_dict['g_ema']['style_affines_conv.0.modulation.weight']= model_dict['g_ema'][key]
                del model_dict['g_ema'][key]
              else:
                model_dict['g_ema']['style_affines_conv.'+str(int(key[6:8].strip('.'))+1)+'.modulation.weight']= model_dict['g_ema'][key]
                del model_dict['g_ema'][key]
          if key[-15:]=='modulation.bias':
            if 'rgb' in key:
              if key[6] == '1':
                model_dict['g_ema']['style_affines_rgb.0.modulation.bias']= model_dict['g_ema'][key]
                del model_dict['g_ema'][key]
              else:
                model_dict['g_ema']['style_affines_rgb.'+str(int(key[8:10].strip('.'))+1)+'.modulation.bias']= model_dict['g_ema'][key]
                del model_dict['g_ema'][key]
            else:
              if key[4] == '1':
                model_dict['g_ema']['style_affines_conv.0.modulation.bias']= model_dict['g_ema'][key]
                del model_dict['g_ema'][key]
              else:
                model_dict['g_ema']['style_affines_conv.'+str(int(key[6:8].strip('.'))+1)+'.modulation.bias']= model_dict['g_ema'][key]
                del model_dict['g_ema'][key]
        del tmp
    return model_dict

def generate_latents_mapping(num_latents, generator, latent_shape = 512):
    """Generates latents for stylegan2 models using the mapping network """
    latents = []
    max_batch = num_latents//4
    for i in tqdm.tqdm(range(max_batch), leave=False):
        latent = generator.get_latent(torch.randn(4, latent_shape, device=device))
        latents.append(latent.cpu().detach())
        del latent
    return torch.concat(latents)


def generate_latents_encoder(num_latents, loader, classifier, encoder):
    """Generates latents for the stylex models using the encoder network and the classifier"""
    latents = []
    encoder.eval()
    max_batch = num_latents//4
    for i, batch in tqdm.tqdm(enumerate(loader), total=max_batch, leave=False):
        if i >= max_batch:
            break
        batch = torch.movedim(batch, 3, 1).to(device)
        encoded = encoder(batch)
        classifier.eval()
        batch_classes = classifier(batch)
        latent = torch.concat((encoded, batch_classes),dim=1)
        latents.append(latent.cpu().detach())
        del batch
        del encoded
        del batch_classes
        del latent
    return torch.concat(latents)

def process_latents(dlatents, generator, classifier):
    """Returns expanded latents usable with the generator,
    the classifier probabilities, and the stylevectors for a list of dlatents"""
    expanded_latents = []
    all_style_vectors = []
    base_probs = []
    num_layers = len(generator.style_affines_conv)
    for i in tqdm.tqdm(range(len(dlatents)), leave=False):
        expanded_dlatent_tmp = torch.tile(
              torch.unsqueeze(dlatents[i], 0),
              [1, num_layers, 1]).to(device)

        styles = generator.get_styles(expanded_dlatent_tmp.to(device))
        imgs,_, _ = generator(styles=styles, input_is_styles=True)
        classifier.eval()
        base_probs.extend(classifier(imgs).cpu().detach().numpy())
        all_style_vectors.extend(torch.cat(styles[0], dim=1).cpu().detach().numpy())
        expanded_latents.extend(expanded_dlatent_tmp.unsqueeze(0))
    expanded_latents = torch.concat(expanded_latents, dim=0)
    return expanded_latents, base_probs, all_style_vectors

def find_fliprate(expanded_latents: torch.Tensor,
                  base_probs: np.ndarray, 
                  s_indices_and_signs: list,
                  style_min: np.ndarray,
                  style_max: np.ndarray,
                  generator: torch.nn.Module,
                  classifier: torch.nn.Module,
                  top_k: int = 10,
                  shift_size: float = 1):
    """Returns the images that have their classification flipped
    given a list of stylespace coordinates and directions.

    Args:
    expanded_latents: a tensor of expanded image latents, with shape [num_images, num_layers, latent_size].
    base_probs: array of classifier logits for the images. 
    s_indices_and_signs: list of tuples with a direction and stylespace index.
    style_min: array with the minimum values for every stylespace coordinate.
    style_max: array with the maximum values for every stylespace coordinate.
    generator: the generator model.
    classifier: the classifier model.
    top_k: amount of stylespace indices to change.
    shift_size: factor of the shift of the style indices.
    """
    
    flipped = []
    for i, dlatent in tqdm.tqdm(enumerate(expanded_latents), total=expanded_latents.shape[0], leave=False):
        generator.eval()
        network_inputs = generator.get_styles(dlatent.unsqueeze(0))
        style_vector = torch.concat(
            generator.get_styles(dlatent.unsqueeze(0))[0],
            dim=1).cpu().detach().numpy()
        base_class = np.argmax(base_probs[i])

        # create changed style vector
        for direction, sindex in s_indices_and_signs[:top_k]:
            direction = base_class == direction            
            orig_value = style_vector[0, sindex]
            target_value = (style_min[sindex] if direction == 0 else style_max[sindex])
            weight_shift = shift_size * (target_value - orig_value)

            layer_idx, in_idx = sindex_to_layer_idx_and_index(generator,  sindex)
            layer_one_hot = torch.unsqueeze(
              torch.nn.functional.one_hot(torch.tensor(in_idx).to(device).to(torch.int64),
                                          network_inputs[0][layer_idx].shape[1]), 0)
            network_inputs[0][layer_idx] += (weight_shift * layer_one_hot)
        
        generator.eval()    
        images_out, _, _ =  generator(styles=network_inputs, input_is_styles=True)
        images_out = torch.clamp(images_out, -1, 1)
        classifier.eval()
        result = classifier(images_out)
        change_prob = torch.nn.functional.softmax(result, dim=1).cpu().detach().numpy()[0]
        if np.argmax(change_prob) != np.argmax(base_probs[i]):
            flipped.append(i)
    return flipped
