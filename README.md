# PCE-Palm (AAAI-24)
*PCE-Palm: Palm Crease Energy based Two-stage Realistic Pseudo-palmprint Generation*
| [Paper](https://ojs.aaai.org/index.php/AAAI/article/view/28039)

## Example results
<img src='imgs/pce-image.png' width=820>  


## Prerequisites
- Python 3
- NVIDIA GPU + CUDA CuDNN

## Getting Started ###

### Installation
- Clone this repo:
```bash
git clone https://github.com/Ukuer/PCE-Palm.git
cd PCE-Palm
``` 
- Install dependencies:
`
pip install -r requirements.txt
`
- More details:
This code borrows heavily from the [RPG-Palm](https://github.com/Ukuer/RPG-Palm) repository. 
You can find more details about the original code in the [RPG-Palm](https://github.com/Ukuer/RPG-Palm).

### Use a Pre-trained Model
- Download [pce-checkpoints](https://drive.google.com/file/d/1r_1vdrVaqrBjuBktBKaj5fEwIXzbka8s/view?usp=sharing), unzip it and place it in `./checkpoints `.
- Download [CUT-checkpoints](https://drive.google.com/file/d/1epH7GV3g9fk4_RwOX8x-uMo5iKOlj0I4/view?usp=sharing), unzip it and place it in `./CUT/checkpoints `.

- Then `bash ./inference.sh`. Noted that you should modify some contents in `./inference.sh` to meet you requirements.

## Model Training
### Tools
- The proposed PCEM can be found in `./PCEM_numpy.py`. You can use it to get the PCE images from palmprint ROIs.
- The propsoed LFEB can be found in `./LFEM_pytorch.py`. You can add it in your network.
- The improved bezier curves can be found in `./syn_bezier.py`. 

### Training 
Our proposed method is a two-stage method. 
The first stage is train a modified CUT model with bezier curves images and PCE images. 
The second stage is to train a generation model with paired PCE images and real palmprints.

- To train a modified CUT model:
    - Firstly, extract PCE images from palmprint ROIs using `./PCEM_numpy.py`.
    - Then, genrate bezier curves images using `./syn_bezier.py`, with a equal number of PCE images.
    - Finally, train a modified CUT model. You can find more details from [CUT](https://github.com/taesungp/contrastive-unpaired-translation.git) origin repository.
    - Noted that set `--netG resnet_9blocks_lfeb`.

- To train a generation model:
    - Train the model with paired PCE images and real palmprints.
    - Then, `bash run.sh`.
    - To view training results and loss plots, run `python -m visdom.server` and click the URL http://localhost:8097. To see more intermediate results, check out  `./checkpoints/NAME/web/index.html`. See [RPG-Palm](https://github.com/Ukuer/RPG-Palm) for more details.
    - Noted that we use the augmentation module from [Stylegan2-ADA](https://github.com/NVlabs/stylegan2-ada). If you have any dependencies issues, please refer to the [Stylegan2-ADA](https://github.com/NVlabs/stylegan2-ada) repository.

### Citation

If you find this useful for your research, please use the following.

```
@inproceedings{jin2024pce,
  title={PCE-Palm: Palm Crease Energy Based Two-Stage Realistic Pseudo-Palmprint Generation},
  author={Jin, Jianlong and Shen, Lei and Zhang, Ruixin and Zhao, Chenglong and Jin, Ge and Zhang, Jingyun and Ding, Shouhong and Zhao, Yang and Jia, Wei},
  booktitle={Proceedings of the AAAI Conference on Artificial Intelligence},
  volume={38},
  number={3},
  pages={2616--2624},
  year={2024}
}

@inproceedings{shen2023rpg,
  title={RPG-Palm: Realistic Pseudo-data Generation for Palmprint Recognition},
  author={Shen, Lei and Jin, Jianlong and Zhang, Ruixin and Li, Huaen and Zhao, Kai and Zhang, Yingyi and Zhang, Jingyun and Ding, Shouhong and Zhao, Yang and Jia, Wei},
  booktitle={Proceedings of the IEEE/CVF International Conference on Computer Vision},
  pages={19605--19616},
  year={2023}
}
```

If you have any questions or encounter any issues with the this code, please feel free to contact me (email: jianlong@mail.hfut.edu.cn). 
I would be more than happy to assist you in any way I can.

### Acknowledgements

This code borrows heavily from the [RPG-Palm](https://github.com/Ukuer/RPG-Palm) repository, [CUT](https://github.com/taesungp/contrastive-unpaired-translation.git) repository, [BicycleGAN](https://github.com/junyanz/BicycleGAN/tree/master) repository, and [Stylegan2-ADA](https://github.com/NVlabs/stylegan2-ada).
