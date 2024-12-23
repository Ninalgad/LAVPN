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
import torchvision
import torch.nn as nn
from utils import *
import torch.nn.functional as F

INPUT_PATH = Path("/input")
OUTPUT_PATH = Path("/output")


def get_default_device():
    if torch.cuda.is_available():
        print("Using gpu device")
        return torch.device('cuda')
    else:
        print("Using cpu device")
        return torch.device('cpu')


def point_wise_feed_forward_network(d_in, d_out, dff):
    return nn.Sequential(
        nn.Linear(d_in, dff),
        nn.ReLU(),
        nn.Linear(dff, dff),
        nn.ReLU(),
        nn.Linear(dff, d_out)
    )


class ClassifierModel(nn.Module):
    def __init__(self, pretrained=True):
        super(ClassifierModel, self).__init__()
        weights = None
        if pretrained:
            weights = 'IMAGENET1K_V1'
        self.encoder = torchvision.models.vgg16(weights=weights).features
        self.head = point_wise_feed_forward_network(512, 1, 128)

    def forward(self, x):
        x = self.encoder(x)

        # global max pool
        x = F.max_pool2d(x, kernel_size=x.size()[2:])
        x = torch.squeeze(x, dim=(2, 3))

        return self.head(x)


def run():
    # Read the input
    adc_ss = load_image_file_as_array(
        location=INPUT_PATH / "images/skull-stripped-adc-brain-mri",
    )
    z_adc = load_image_file_as_array(
        location=INPUT_PATH / "images/z-score-adc",
    )
    inp = create_input_array(z_adc, adc_ss, 256, channels_first=True)  # (n, 3 256, 256)
    # Process the inputs: any way you'd like
    _show_torch_cuda_info()
    device = get_default_device()
    paths = [
        "outcomes-model-11254.pt", "outcomes-model-19529.pt", "outcomes-model-660969.pt",
        "outcomes-model-682359.pt", "outcomes-model-944415.pt", "outcomes-model-2150913.pt",
        "outcomes-model-47554755.pt", "outcomes-model-76607660.pt", "outcomes-model-94835948.pt",
        "outcomes-model-4618746.pt"
    ]

    with torch.no_grad():
        inp = torch.from_numpy(inp).to(device)

        model = ClassifierModel(pretrained=False)
        model = model.to(device)

        ensemble_prediction = 0
        for path in paths:
            ckpt = torch.load(path, device)
            model.load_state_dict(ckpt["model_state_dict"])
            thresh = ckpt["best_thresh"]
            del ckpt

            out = F.sigmoid(model(inp)).detach().cpu().numpy()
            out = np.max(out)
            out = (out > thresh).astype(int)

            ensemble_prediction += out
            del out

        thresh = max(1, 1 + int(len(paths) / 2))
        output_2_year_neurocognitive_outcome = (ensemble_prediction >= thresh).astype(np.uint8)

    # Save your output
    write_json_file(
        location=OUTPUT_PATH / "2-year-neurocognitive-outcome.json",
        content=int(output_2_year_neurocognitive_outcome)
    )

    return 0


def write_json_file(*, location, content):
    # Writes a json file
    with open(location, 'w') as f:
        f.write(json.dumps(content, indent=4))


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
