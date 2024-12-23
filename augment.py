import albumentations as albu


def get_transform(inp_size):
  return albu.Compose([
      albu.HorizontalFlip(),
      albu.VerticalFlip(),
      albu.Resize(inp_size, inp_size, always_apply=True),
  ])