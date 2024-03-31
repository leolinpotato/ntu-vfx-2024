# NTU-VFX2024 HW1

## Member
B10902024 林宸宇
B11902174 陳璿修

## File Structure
[hw1_17]
|-- [data]
|---- [images]
|---- [HDR_images]
|---- [tone_mapped_images]
|-- [code]
|---- main.py
|---- HDR.py
|---- JBF.py
|---- Reinhard.py
|---- alignment.py
|-- README.md
|-- report.pdf
|-- result.png

## Dependency
- python==3.10.0
- contourpy==1.2.0
- cycler==0.12.1
- ExifRead==3.0.0
- fonttools==4.50.0
- kiwisolver==1.4.5
- matplotlib==3.8.3
- natsort==8.4.0
- numpy==1.26.4
- opencv-contrib-python==4.9.0.80
- packaging==24.0
- pillow==10.2.0
- pyparsing==3.1.2
- python-dateutil==2.9.0.post0
- six==1.16.0

## Run
```
cd code
python main.py -a -t all -g
```
Arguments:
- \-a: align the images
- \-t: set the tone-mapping algorithm, can be set to
	- Reinhard
	- Bilateral
	- OpenCVReinhard
	- OpenCVDrago
	- OpenCVMantiuk
	- all
- \-g: perform ghost removal
- \-p: plot HDR image and response curve 

