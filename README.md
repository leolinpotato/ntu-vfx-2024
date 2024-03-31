# NTU-VFX2024

# file structure
hw1  
|- image  
| |- align  
| | |- 'imagename'_align.jpg * 10  
| |- 'imagename'.jpg * 10  
|- result  
|- alignment.py  
|- HDR.py  
|- JBF.py
|- Reinhard.py
|- main.py


# change log
## 2024/3/29 14:34
- make alignment faster
- change aligned image name into 'imagename'_align.jpg
- aligned image will be saved into hw1/image/align directory 
- add padding to image

## 2024/3/30 21:14
- use reflect mode for padding

## 2024/3/30 23:16
- add tone-mapping feature using Photographic Tone Reproduction

## 2024/3/31 14:14
- add Bilateral filter
- rearrange file structure
- add OpenCV Reinhard
