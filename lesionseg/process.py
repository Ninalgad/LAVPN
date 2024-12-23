"""
The following is a simple example algorithm.

It is meant to run within a container.

To run it locally, you can call the following bash script:

  ./test.sh

This will start the inference and reads from ./test/input and outputs to ./test/output

To save the container and prep it for upload to Grand-Challenge.org you can call:

  ./export.sh

Any container that shows the same behavior will do, this is purely an example of how one COULD do it.

Happy programming!
"""
from pathlib import Path
import json
from glob import glob
import SimpleITK
import numpy as np
import torch
import os
import torch.nn as nn
from utils import create_input_array
from scipy import ndimage


INPUT_PATH = Path("/input")
OUTPUT_PATH = Path("/output")


def get_default_device():
    if torch.cuda.is_available():
        print("Using gpu device")
        return torch.device('cuda')
    else:
        print("Using cpu device")
        return torch.device('cpu')


class SegNet(nn.Module):
    def __init__(self, output_dim=1, hidden_dim=64, pretrained=True):
        super().__init__()
        from ternausnet.models import UNet16
        self.encoder = UNet16(num_classes=hidden_dim, pretrained=pretrained)
        self.hidden_layer = nn.Conv2d(hidden_dim, hidden_dim, 1)
        self.output_layer = nn.Conv2d(hidden_dim, output_dim, 1)

    def forward(self, x):
        x = self.encoder(x)
        x = torch.nn.ReLU()(x)
        x = self.hidden_layer(x)
        x = torch.nn.ReLU()(x)
        return self.output_layer(x)


def aug_predict(model, inp, device):
    # x: (n, 3, d, d)
    prediction = 0
    x = torch.tensor(inp, dtype=torch.float32).to(device)
    prediction += nn.functional.sigmoid(model(x)).detach().cpu().numpy()

    p = nn.functional.sigmoid(model(torch.flip(x, [-1])))
    prediction += torch.flip(p, [-1]).detach().cpu().numpy()

    p = nn.functional.sigmoid(model(torch.flip(x, [-2])))
    prediction += torch.flip(p, [-2]).detach().cpu().numpy()

    p = nn.functional.sigmoid(model(torch.flip(x, [-1, -2])))
    prediction += torch.flip(p, [-1, -2]).detach().cpu().numpy()

    return prediction / 4.


def run():
    # Read the input
    adc_ss = load_image_file_as_array(
        location=INPUT_PATH / "images/skull-stripped-adc-brain-mri",
    )
    z_adc = load_image_file_as_array(
        location=INPUT_PATH / "images/z-score-adc",
    )
    num, sx, sy = adc_ss.shape

    inp = create_input_array(z_adc, adc_ss, 256, channels_first=True)  # (n, 3 256, 256)
    device = get_default_device()
    paths = [
        "model-2150913.pt", "model-944415.pt", "model-682359.pt", "model-660969.pt", "model-19529.pt",
        "model-11254.pt", "model-94835948.pt", "model-47554755.pt", "model-4618746.pt"
    ]

    # Process the inputs: any way you'd like
    _show_torch_cuda_info()

    with torch.no_grad():
        inp = torch.from_numpy(inp).to(device)

        model = SegNet(pretrained=False)
        model = model.to(device)

        ensemble_prediction = 0
        for path in paths:
            ckpt = torch.load(path, device)
            model.load_state_dict(ckpt["model_state_dict"])
            thresh = ckpt["best_thresh"]
            del ckpt

            # out = aug_predict(model, inp, device)
            x = torch.tensor(inp, dtype=torch.float32).to(device)
            out = nn.functional.sigmoid(model(x)).detach().cpu().numpy()

            out = np.squeeze(out, axis=1)
            out = ndimage.zoom(out, (1, sx / 256, sy / 256))
            out = (out > thresh).astype(np.uint8)

            assert out.shape == (num, sx, sy)
            ensemble_prediction = ensemble_prediction + out
            del out

        thresh = max(1, int(len(paths)/2))
        ensemble_prediction = (ensemble_prediction >= thresh).astype(np.uint8)

    hie_segmentation = SimpleITK.GetImageFromArray(ensemble_prediction)

    # For now, let us save predictions
    save_image(hie_segmentation)

    return 0


def write_json_file(*, location, content):
    # Writes a json file
    with open(location, 'w') as f:
        f.write(json.dumps(content, indent=4))


def save_image(pred_lesion):
    relative_path = "images/hie-lesion-segmentation"
    output_directory = OUTPUT_PATH / relative_path

    output_directory.mkdir(exist_ok=True, parents=True)

    file_save_name = output_directory / "overlay.mha"
    print(file_save_name)

    SimpleITK.WriteImage(pred_lesion, file_save_name)
    check_file = os.path.isfile(file_save_name)
    print("check file", check_file)


def load_image_file_as_array(*, location):
    # Use SimpleITK to read a file
    input_files = glob(str(location / "*.mha"))
    result = SimpleITK.ReadImage(input_files[0])

    # Convert it to a Numpy array
    return SimpleITK.GetArrayFromImage(result)


def _show_torch_cuda_info():
    print("=+=" * 10)
    print("Collecting Torch CUDA information")
    print(f"Torch CUDA is available: {(available := torch.cuda.is_available())}")
    if available:
        print(f"\tnumber of devices: {torch.cuda.device_count()}")
        print(f"\tcurrent device: {(current_device := torch.cuda.current_device())}")
        print(f"\tproperties: {torch.cuda.get_device_properties(current_device)}")
    print("=+=" * 10)


if __name__ == "__main__":
    raise SystemExit(run())
