## Overview
1. This system works on 2 levels of noise reduction from live streaming of audio from microphones: Single, Multi
2. For single speakers mode there are 2 scenarios:  a. when there are more than 1 speaker b. just 1 speaker. In the 1st case we will just take the dominant speaker speech parts from the stream & save in the output file (we are using energy/intensity based method for dominance). For 2nd case, we will just denoise the background noises.
3. For multi speakers as well we will just denoise background noises.
4. For these works we will be using speaker diarizations with pyannote.audio & denoising with DeepFilternet libraries which both can be used with very low latency & edge devices with cpu only. Though the performance enhances with the 'cuda'.

## Instllation:
1. Firstly create a virtual environment in your local windows11 system.
```bash
pip install virtualenv
python -m venv audenv
.\audenv\Scripts\activate
```
2. Now install libraries:
```bash
pip install -r requirements.txt
```
3. Now we can just start the system by:
```bash
python main.py
```
And it will ask how many speaker mode you want: single or multi.

4. Alternatively we can run the cells of AudioDenoiser.ipynb one by one.

5. Limitations: It is not tested on the overlapping voices that much so some cases it will not provide expected results.
