## 3DDFAV2 FACE FILTERING

## Prerequisites

- Run: `pip install -r requirements.txt`

## Usage
- Clone this repo 
```shell
    git clone https://github.com/PXThanhLam/FaceFilter_3ddfav2
    cd FaceFilter_3ddfav2
```
- Build the cython version of NMS, Sim3DR, and the faster mesh render
```shell
    sh ./build.sh
``` 
- Run demos
```shell
    python test_tkinter2.py
    # Using onnx (faster)
    python test_tkinter2.py --onnx True
``` 