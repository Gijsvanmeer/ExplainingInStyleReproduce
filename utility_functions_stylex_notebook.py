from typing import Optional, Tuple, List
import six.moves.cPickle as cPickle
import tensorflow as tf
import numpy as np
import requests
import tqdm
import collections
import os
import zipfile
from matplotlib import pyplot as plt
from PIL import Image
from PIL import ImageDraw
from PIL import ImageFont
from io import BytesIO
import IPython.display
import torch
from torchvision import models
from torch.utils.data import Dataset, DataLoader

device = 'cuda' if torch.cuda.is_available() else 'cpu'

def make_animation(image: torch.Tensor,
                   resolution: int,
                   figsize: Tuple[int, int] = (20, 8)):
  fig = plt.figure(1, figsize=figsize)
  _ = plt.gca()
  image = image.cpu().detach().numpy().astype(np.uint8)
  def transpose_image(image):
    image_reshape = image.reshape([-1, resolution, resolution, 3])
    return image_reshape.transpose([1, 0, 2, 3]).reshape([resolution, -1, 3])
  im = plt.imshow(transpose_image(image[:, :resolution, :]),
                  interpolation='none')
  def animate_func(i):
    im.set_array(transpose_image(image[:, resolution*i:resolution*(i+1), :]))
    return [im]

  animation = matplotlib.animation.FuncAnimation(
      fig, animate_func, frames=image.shape[1] // resolution, interval=600)

  plt.close(1)
  return animation


def show_image(image, fmt='png'):
  image = image.cpu().detach()
  if image.dtype == torch.int:
    image = np.uint8(image)
  elif image.dtype == torch.float:
    image = np.uint8(image * 127.5 + 127.5)
  if image.shape[0] == 3:
    image = np.transpose(image, (1, 2, 0))
  bytes_io = BytesIO()
  Image.fromarray(image).save(bytes_io, fmt)
  IPython.display.display(IPython.display.Image(data=bytes_io.getvalue()))


def filter_unstable_images(style_change_effect: np.ndarray,
                           effect_threshold: float = 0.3,
                           num_indices_threshold: int = 750) -> np.ndarray:
  """Filters out images which are affected by too many S values."""
  unstable_images = (
      np.sum(np.abs(style_change_effect) > effect_threshold, axis=(1, 2, 3)) >
      num_indices_threshold)
  style_change_effect[unstable_images] = 0
  return style_change_effect

def call_synthesis(generator: torch.nn.Module,
                   dlatents_in: torch.Tensor,
                   conditioning_in: Optional[torch.Tensor] = None,
                   labels_in: Optional[torch.Tensor] = None,
                   training: bool = False,
                   num_layers: int = 14,
                   dlatent_size: int = 512,
                   input_is_styles: bool = False,
                   styles: Optional[torch.Tensor] = None) -> torch.Tensor:
  """Calls the synthesis.

  Args:
    dlatents_in: the intermediate latent representation of shape [batch size,
      num_layers, dlatent_size].
    conditioning_in: Conditioning input to the synthesis network (can be an
      image or output from an encoder) of shape [minibatch, channels,
      resolution, resolution]. Set to None if unused.
    labels_in: of shape [batch_size, label_size]. Set to None if unused.
    training: Whether this is a training call.

  Returns:
    The output images and optional latent vector.

  """
  if labels_in is not None:
    zero_labels = torch.zeros_like(labels_in)
    dlatents_labels = torch.tile(torch.unsqueeze(labels, 1), [1, num_layers, 1])
    if dlatent_size > 0:
      dlatents_expanded = torch.concat([dlatents_in, dlatents_labels], axis=2)
    else:
      dlatents_expanded = dlatents_labels
  else:
    if dlatent_size == 0:
      raise ValueError('Dlatents are empty and no labels were provided.')
    dlatents_expanded = dlatents_in
  # Evaluate synthesis network.
  generator.train(training)
  style_vector_blocks, style_vector_torgb = generator.get_styles(dlatents_expanded)
  if conditioning_in is not None:
    network_inputs = (style_vector_blocks, style_vector_torgb,
                      conditioning_in)
  else:
    network_inputs = (style_vector_blocks, style_vector_torgb)

  if input_is_styles:
    network_inputs = styles
  
  synthesis_results = generator(styles=network_inputs, input_is_styles=True)[0]

  # Return requested outputs.
  return torch.clamp(synthesis_results, -1, 1)


def discriminator_filter(style_change_effect: np.ndarray,
                         all_dlatents: np.ndarray,
                         generator: torch.nn.Module,
                         discriminator: torch.nn.Module,
                         classifier: torch.nn.Module,
                         sindex: int,
                         style_min: float,
                         style_max: float,
                         class_index: int,
                         num_images: int = 10,
                         label_size: int = 2,
                         change_threshold: float = 0.5,
                         shift_size: float = 2,
                         effect_threshold: float = 0.2,
                         sindex_offset: int = 0) -> bool:
  """Returns false if changing the style index adds artifacts to the images.

  Args:
    style_change_effect: A shape of [num_images, 2, style_size, num_classes].
      The effect of each change of style on specific direction on each image.
    all_dlatents: The dlatents of each image, shape of [num_images,
      dlatent_size].
    generator: The generator model. Either StyleGAN or GLO.
    discriminator: The discriminator model.
    sindex: The style index.
    style_min: The style min value in all images.
    style_max: The style max value in all images.
    class_index: The index of the class to check.
    num_images: The number of images to do the disciminator_filter test.
    label_size: The label size.
    change_threshold: The maximal change allowed in the discriminator
      prediction.
    shift_size: The size to shift the style index.
    effect_threshold: Used for choosing images that the classification was
      changed enough.
    sindex_offset: The offset of the style index if style_change_effect contains
      some of the layers and not all styles.
  """
  for style_sign_index in range(2):
    images_idx = ((style_change_effect[:, style_sign_index, sindex,
                                       class_index]) >
                  effect_threshold).nonzero()[0]

    images_idx = images_idx[:num_images]
    dlatents = all_dlatents[images_idx]

    for i in range(len(images_idx)):
      cur_dlatent = dlatents[i:i + 1]
      (discriminator_orig, 
       discriminator_change) = get_discriminator_results_given_dlatent(
           dlatent=cur_dlatent,
           generator=generator,
           discriminator=discriminator,
           classifier=classifier,
           class_index=class_index,
           sindex=sindex + sindex_offset,
           s_style_min=style_min,
           s_style_max=style_max,
           style_direction_index=style_sign_index,
           shift_size=shift_size,
           label_size=label_size)

      if np.abs((discriminator_orig - discriminator_change).cpu().detach()) > change_threshold:
        return False
  return True


def find_significant_styles_image_fraction(
    style_change_effect: np.ndarray,
    num_indices: int,
    class_index: int,
    generator: torch.nn.Module,
    classifier: torch.nn.Module,
    all_dlatents: np.ndarray,
    style_min: np.ndarray,
    style_max: np.ndarray,
    effect_threshold: float = 0.2,
    min_changed_images_fraction: float = 0.03,
    label_size: int = 2,
    sindex_offset: int = 0,
    discriminator: Optional[torch.nn.Module] = None,
    discriminator_threshold: float = 0.2) -> List[Tuple[int, int]]:
  """Returns indices in the style vector which affect the classifier.

  Args:
    style_change_effect: A shape of [num_images, 2, style_size, num_classes].
      The effect of each change of style on specific direction on each image.
    num_indices: Number of styles in the result.
    class_index: The index of the class to visualize.
    generator: The generator model. Either StyleGAN or GLO.
    all_dlatents: The dlatents of each image, shape of [num_images,
      dlatent_size].
    style_min: An array with the min value for each style index.
    style_max: An array with the max value for each style index.
    effect_threshold: Minimal change of classifier output to be considered.
    min_changed_images_fraction: Minimal fraction of images which are changed.
    label_size: The label size.
    sindex_offset: The offset of the style index if style_change_effect contains
      some of the layers and not all styles.
    discriminator: The discriminator model. If None, don't filter style indices.
    discriminator_threshold: Used in discriminator_filter to define the maximal
      change allowed in the discriminator prediction.
    
  """
  effect_positive = np.sum(
      style_change_effect[:, :, :, class_index] > effect_threshold, axis=0)
  effect_positive = effect_positive.flatten()
  all_sindices = []
  sindices = np.argsort(effect_positive)[::-1]
  if discriminator is not None:
    print('Using discriminator...')
  for sindex in sindices[:num_indices*2]:
    if (effect_positive[sindex] <
        min_changed_images_fraction * style_change_effect.shape[0]):
      break
    if discriminator is not None:
      s_index = sindex % style_change_effect.shape[2]
      if not discriminator_filter(
          style_change_effect,
          all_dlatents,
          generator,
          discriminator,
          classifier,
          s_index,
          style_min[s_index + sindex_offset],
          style_max[s_index + sindex_offset],
          class_index,
          label_size=label_size,
          change_threshold=discriminator_threshold,
          sindex_offset=sindex_offset):
        continue
    all_sindices.append(sindex)
    if len(all_sindices) == num_indices:
      break

  return [(x // style_change_effect.shape[2],
           (x % style_change_effect.shape[2]) + sindex_offset)
          for x in all_sindices]


def find_significant_styles(
    style_change_effect: np.ndarray,
    num_indices: int,
    class_index: int,
    discriminator: Optional[torch.nn.Module],
    generator: torch.nn.Module,
    classifier: torch.nn.Module,
    all_dlatents: np.ndarray,
    style_min: np.ndarray,
    style_max: np.ndarray,
    max_image_effect: float = 0.2,
    label_size: int = 2,
    discriminator_threshold: float = 0.2,
    sindex_offset: int = 0) -> List[Tuple[int, int]]:
  """Returns indices in the style vector which affect the classifier.

  Args:
    style_change_effect: A shape of [num_images, 2, style_size, num_classes].
      The effect of each change of style on specific direction on each image.
    num_indices: Number of styles in the result.
    class_index: The index of the class to visualize.
    discriminator: The discriminator model. If None, don't filter style indices.
    generator: The generator model. Either StyleGAN or GLO.
    all_dlatents: The dlatents of each image, shape of [num_images,
      dlatent_size].
    style_min: An array with the min value for each style index.
    style_max: An array with the max value for each style index.
    max_image_effect: Ignore contributions of styles if the previously found
      styles changed the probability of the image by more than this threshold.
    label_size: The label size.
    discriminator_threshold: Used in discriminator_filter to define the maximal
      change allowed in the discriminator prediction.
    sindex_offset: The offset of the style index if style_change_effect contains
      some of the layers and not all styles.
  """

  num_images = style_change_effect.shape[0]
  style_effect_direction = np.maximum(
      0, style_change_effect[:, :, :, class_index].reshape((num_images, -1)))

  images_effect = np.zeros(num_images)
  all_sindices = []
  discriminator_removed = []
  while len(all_sindices) < num_indices:
    next_s = np.argmax(
        np.mean(
            style_effect_direction[images_effect < max_image_effect], axis=0))
    if discriminator is not None:
      sindex = next_s % style_change_effect.shape[2]
      if sindex == 0:
        break
      if not discriminator_filter(
          style_change_effect=style_change_effect,
          all_dlatents=all_dlatents,
          generator=generator,
          discriminator=discriminator,
          classifier=classifier,
          sindex=sindex,
          style_min=style_min[sindex + sindex_offset],
          style_max=style_max[sindex + sindex_offset],
          class_index=class_index,
          label_size=label_size,
          change_threshold=discriminator_threshold,
          sindex_offset=sindex_offset):
        style_effect_direction[:, next_s] = np.zeros(num_images)
        discriminator_removed.append(sindex)
        continue

    all_sindices.append(next_s)
    images_effect += style_effect_direction[:, next_s]
    style_effect_direction[:, next_s] = 0
  return [(x // style_change_effect.shape[2],
           (x % style_change_effect.shape[2]) + sindex_offset)
          for x in all_sindices]


def _float_features(values):
  """Returns a float_list from a float / double."""
  return tf.train.Feature(float_list=tf.train.FloatList(value=values))





def sindex_to_layer_idx_and_index(generator: torch.nn.Module,
                                  sindex: int) -> Tuple[int, int]:
  layer_shapes = []
  for dense in generator.style_affines_conv._modules.values():
    layer_shapes.append(dense.modulation.bias.size(0))
  layer_shapes_cumsum = np.concatenate([[0], np.cumsum(layer_shapes)])
  layer_idx = (layer_shapes_cumsum <= sindex).nonzero()[0][-1]
  return layer_idx, sindex - layer_shapes_cumsum[layer_idx]


def get_classifier_results(generator: torch.nn.Module, 
                           expanded_dlatent: torch.Tensor,
                           use_softmax: bool = False,
                           batch_size=1,
                           input_is_styles: bool = False,
                           styles: Optional[torch.Tensor] = None):
  img = call_synthesis(generator, expanded_dlatent, 
                       input_is_styles=input_is_styles, styles=styles)
  classifier.eval()
  results = classifier(img).cpu().detach()
  if use_softmax:
    results =  torch.nn.functional.softmax(results).numpy()
  else:
    results = results.numpy()

  if batch_size == 1:
    return results[0]
  return results


def draw_on_image(image: np.ndarray, number: float,
                  font_file: str,
                  font_fill: Tuple[int, int, int] = (0, 0, 255)) -> np.ndarray:
  """Draws a number on the top left corner of the image."""
  fnt = ImageFont.truetype(font_file, 20)
  image = image.cpu().detach().numpy()
  out_image = Image.fromarray((image * 127.5 + 127.5).astype(np.uint8))
  draw = ImageDraw.Draw(out_image)
  draw.multiline_text((10, 10), ('%.3f' % number), font=fnt, fill=font_fill)
  return torch.tensor(np.array(out_image)).to(device)


def generate_change_image_given_dlatent(
    dlatent: np.ndarray,
    generator: torch.nn.Module,
    classifier: Optional[torch.nn.Module],
    class_index: int,
    sindex: int,
    s_style_min: float,
    s_style_max: float,
    style_direction_index: int,
    shift_size: float,
    label_size: int = 2,
) -> Tuple[np.ndarray, float, float]:
  """Modifies an image given the dlatent on a specific S-index.

  Args:
    dlatent: The image dlatent, with sape [dlatent_size].
    generator: The generator model. Either StyleGAN or GLO.
    classifier: The classifier to visualize.
    class_index: The index of the class to visualize.
    sindex: The specific style index to visualize.
    s_style_min: The minimal value of the style index.
    s_style_max: The maximal value of the style index.
    style_direction_index: If 0 move s to it's min value otherwise to it's max
      value.
    shift_size: Factor of the shift of the style vector.
    label_size: The size of the label.

  Returns:
    The image after the style index modification, and the output of
    the classifier on this image.
  """
  num_layers = len(generator.style_affines_conv)
  expanded_dlatent = torch.tile(
          torch.unsqueeze(dlatent, 1),
          [1, num_layers, 1])
  network_inputs = generator.get_styles(expanded_dlatent)
  generator.eval()
  style_vector = torch.concat(
        generator.get_styles(expanded_dlatent)[0],
        dim=1).cpu().detach().numpy()
  orig_value = style_vector[0, sindex]
  target_value = (s_style_min if style_direction_index == 0 else s_style_max)

  weight_shift = shift_size * (target_value - orig_value)

  layer_idx, in_idx = sindex_to_layer_idx_and_index(generator,  sindex)
  layer_one_hot = torch.unsqueeze(
      torch.nn.functional.one_hot(torch.tensor(in_idx).to(device).to(torch.int64), network_inputs[0][layer_idx].shape[1]), 0)
  network_inputs[0][layer_idx] += (weight_shift * layer_one_hot)
  images_out, _, _ =  generator(styles=network_inputs, input_is_styles=True)
  images_out = torch.clamp(images_out, -1, 1)
  classifier.eval()
  result = classifier(images_out)
  change_prob = torch.nn.functional.softmax(result, dim=1).cpu().detach().numpy()[0, class_index]
  return images_out, change_prob



def get_discriminator_results_given_dlatent(
    dlatent: np.ndarray,
    generator: torch.nn.Module,
    discriminator: torch.nn.Module,
    classifier: torch.nn.Module,
    class_index: int,
    sindex: int,
    s_style_min: float,
    s_style_max: float,
    style_direction_index: int,
    shift_size: float = 2,
    label_size: int = 2,
) -> Tuple[float, float]:
  """Modifies an image given the dlatent on a specific S-index.

  Args:
    dlatent: The image dlatent, with sape [dlatent_size].
    generator: The generator model. Either StyleGAN or GLO.
    class_index: The index of the class to visualize.
    sindex: The specific style index to visualize.
    s_style_min: The minimal value of the style index.
    s_style_max: The maximal value of the style index.
    style_direction_index: If 0 move s to it's min value otherwise to it's max
      value.
    shift_size: Factor of the shift of the style vector.
    label_size: The size of the label.

  Returns:
    The discriminator before and after.
  """
  dlatent = torch.tensor(dlatent).to(device).to(torch.float)
  num_layers = len(generator.style_affines_conv)
  expanded_dlatent = torch.tile(
          torch.unsqueeze(dlatent, 1),
          [1, num_layers, 1])
  network_inputs = generator.get_styles(expanded_dlatent)
  images_out, _, _ =  generator(styles=network_inputs, input_is_styles=True)
  images_out = torch.clamp(images_out, -1, 1)
  labels = torch.tensor(dlatent[:, -label_size:], dtype=torch.float32).to(device)
  discriminator.eval()
  discriminator_before = discriminator(images_out)
  # I am not using the classifier output here, because it is only one.
  change_image, _ = (
      generate_change_image_given_dlatent(dlatent, generator, classifier,
                                          class_index, sindex,
                                          s_style_min, s_style_max,
                                          style_direction_index, shift_size,
                                          label_size))
  labels = torch.nn.functional.softmax(classifier(change_image), dim=1)

  discriminator_after = discriminator(change_image)
  return (discriminator_before, discriminator_after)


def generate_images_given_dlatent(
    dlatent: np.ndarray,
    generator: torch.nn.Module,
    classifier: Optional[torch.nn.Module],
    class_index: int,
    sindex: int,
    s_style_min: float,
    s_style_max: float,
    style_direction_index: int,
    font_file: Optional[str],
    shift_size: float = 2,
    label_size: int = 2,
    draw_results_on_image: bool = True,
    resolution: int = 64,
) -> Tuple[np.ndarray, float, float, float, float]:
  """Modifies an image given the dlatent on a specific S-index.

  Args:
    dlatent: The image dlatent, with sape [dlatent_size].
    generator: The generator model. Either StyleGAN or GLO.
    classifier: The classifier to visualize.
    class_index: The index of the class to visualize.
    sindex: The specific style index to visualize.
    s_style_min: The minimal value of the style index.
    s_style_max: The maximal value of the style index.
    style_direction_index: If 0 move s to it's min value otherwise to it's max
      value.
    font_file: A path to the font file for writing the probability on the image.
    shift_size: Factor of the shift of the style vector.
    label_size: The size of the label.
    draw_results_on_image: Whether to draw the classifier outputs on the images.

  Returns:
    The image before and after the style index modification, and the outputs of
    the classifier before and after the
    modification.
  """
  dlatent = torch.tensor(dlatent).to(device).to(torch.float)
  num_layers = len(generator.style_affines_conv)
  expanded_dlatent = torch.tile(
          torch.unsqueeze(dlatent, 1),
          [1, num_layers, 1])
  network_inputs = generator.get_styles(expanded_dlatent)
  result_image = torch.zeros((resolution, 2 * resolution, 3))
  generator.eval()
  base_image, _, _ = generator(styles=network_inputs, input_is_styles=True)
  base_image = torch.clamp(base_image, -1, 1,)
  classifier.eval()
  result = classifier(base_image)
  base_image = torch.movedim(base_image, 1, 3)
  base_prob = torch.nn.functional.softmax(result, dim=1)[0, class_index]
  if draw_results_on_image:
    result_image[:, :resolution, :] = draw_on_image(
        base_image[0], base_prob, font_file)
  else:
    result_image[:, :resolution, :] = (base_image[0] * 127.5 +
                                       127.5)

  change_image, change_prob = (
      generate_change_image_given_dlatent(dlatent, generator, classifier,
                                          class_index, sindex,
                                          s_style_min, s_style_max,
                                          style_direction_index, shift_size,
                                          label_size))
  change_image = torch.movedim(change_image, 1, 3)
  if draw_results_on_image:
    result_image[:, resolution:, :] = draw_on_image(
        change_image[0], change_prob, font_file)
  else:
    result_image[:, resolution:, :] = (
        torch.clamp(change_image[0], -1, 1) * 127.5 +
                                               127.5)

  return (result_image.to(torch.int), change_prob, base_prob)



def visualize_style(generator: torch.nn.Module,
                    classifier: torch.nn.Module,
                    all_dlatents: np.ndarray,
                    style_change_effect: np.ndarray,
                    style_min: np.ndarray,
                    style_max: np.ndarray,
                    sindex: int,
                    style_direction_index: int,
                    max_images: int,
                    shift_size: float,
                    font_file: str,
                    label_size: int = 2,
                    class_index: int = 0,
                    effect_threshold: float = 0.3,
                    seed: Optional[int] = None,
                    allow_both_directions_change: bool = False,
                    draw_results_on_image: bool = True) -> np.ndarray:
  """Returns an image visualizing the effect of a specific S-index.

  Args:
    generator: The generator model. Either StyleGAN or GLO.
    classifier: The classifier to visualize.
    all_dlatents: An array with shape [num_images, dlatent_size].
    style_change_effect: A shape of [num_images, 2, style_size, num_classes].
      The effect of each change of style on specific direction on each image.
    style_min: The minimal value of each style, with shape [style_size].
    style_max: The maximal value of each style, with shape [style_size].
    sindex: The specific style index to visualize.
    style_direction_index: If 0 move s to its min value otherwise to its max
      value.
    max_images: Maximal number of images to visualize.
    shift_size: Factor of the shift of the style vector.
    font_file: A path to the font file for writing the probability on the image.
    label_size: The size of the label.
    class_index: The index of the class to visualize.
    effect_threshold: Choose images whose effect was at least this number.
    seed: If given, use this as a seed to the random shuffling of the images.
    allow_both_directions_change: Whether to allow both increasing and
      decreasing the classifiaction (used for age).
    draw_results_on_image: Whether to draw the classifier outputs on the images.
  """

  # Choose the dlatent indices to visualize
  if allow_both_directions_change:
    images_idx = (np.abs(style_change_effect[:, style_direction_index, sindex,
                                             class_index]) >
                  effect_threshold).nonzero()[0]
  else:
    images_idx = ((style_change_effect[:, style_direction_index, sindex,
                                       class_index]) >
                  effect_threshold).nonzero()[0]
  if images_idx.size == 0:
    return np.array([])

  if seed is not None:
    np.random.seed(seed)
  np.random.shuffle(images_idx)
  images_idx = images_idx[:min(max_images*10, len(images_idx))]
  dlatents = all_dlatents[images_idx]

  result_images = []
  for i in range(len(images_idx)):
    cur_dlatent = dlatents[i:i + 1]
    (result_image, base_prob, change_prob) = generate_images_given_dlatent(
         dlatent=cur_dlatent,
         generator=generator,
         classifier=classifier,
         class_index=class_index,
         sindex=sindex,
         s_style_min=style_min[sindex],
         s_style_max=style_max[sindex],
         style_direction_index=style_direction_index,
         font_file=font_file,
         shift_size=shift_size,
         label_size=label_size,
         draw_results_on_image=draw_results_on_image)

    if np.abs((change_prob - base_prob).cpu().detach()) < effect_threshold:
      continue
    result_images.append(result_image)
    if len(result_images) == max_images:
      break
  if len(result_images) < 1:
    # No point in returning results with very little images
    return torch.tensor([]).to(device)
  return torch.concat(result_images[:max_images], axis=0)


def visualize_style_by_distance_in_s(
    generator: torch.nn.Module,
    classifier: torch.nn.Module,
    all_dlatents: np.ndarray,
    all_style_vectors_distances: np.ndarray,
    style_min: np.ndarray,
    style_max: np.ndarray,
    sindex: int,
    style_sign_index: int,
    max_images: int,
    shift_size: float,
    font_file: str,
    label_size: int = 2,
    class_index: int = 0,
    draw_results_on_image: bool = True,
    effect_threshold: float = 0.1) -> np.ndarray:
  """Returns an image visualizing the effect of a specific S-index.

  Args:
    generator: The generator model. Either StyleGAN or GLO.
    classifier: The classifier to visualize.
    all_dlatents: An array with shape [num_images, dlatent_size].
    all_style_vectors_distances: A shape of [num_images, style_size, 2].
      The distance each style from the min and max values on each image.
    style_min: The minimal value of each style, with shape [style_size].
    style_max: The maximal value of each style, with shape [style_size].
    sindex: The specific style index to visualize.
    style_sign_index: If 0 move s to its min value otherwise to its max
      value.
    max_images: Maximal number of images to visualize.
    shift_size: Factor of the shift of the style vector.
    font_file: A path to the font file for writing the probability on the image.
    label_size: The size of the label.
    class_index: The index of the class to visualize.
    draw_results_on_image: Whether to draw the classifier outputs on the images.
  """

  # Choose the dlatent indices to visualize
  images_idx = np.argsort(
      all_style_vectors_distances[:, sindex, style_sign_index])[::-1]
  if images_idx.size == 0:
    return np.array([])

  images_idx = images_idx[:min(max_images*10, len(images_idx))]
  dlatents = all_dlatents[images_idx]

  result_images = []
  for i in range(len(images_idx)):
    cur_dlatent = dlatents[i:i + 1]
    (result_image, change_prob, base_prob) = generate_images_given_dlatent(
         dlatent=cur_dlatent,
         generator=generator,
         classifier=classifier,
         class_index=class_index,
         sindex=sindex,
         s_style_min=style_min[sindex],
         s_style_max=style_max[sindex],
         style_direction_index=style_sign_index,
         font_file=font_file,
         shift_size=shift_size,
         label_size=label_size,
         draw_results_on_image=draw_results_on_image)
    if (change_prob - base_prob) < effect_threshold:
      continue
    result_images.append(result_image)


  if len(result_images) < 3:
    # No point in returning results with very little images
    return torch.tensor([]).to(device)
  return torch.concat(result_images[:max_images], dim=0)
