## Multi-style transfer based on the [fast stilization paper]()

To train the network end-to-end only do:

```bash
python train.py
```

...after installing the dependencies. :)

### Docker

The model can be run in a dockerized form either by building it or by downloading it:

The docker image is hosted on `DockerHub` as well:

#### Download

```bash
sudo docker pull qbear666/multi_style
sudo docker tag qbear666/multi_style:latest multi_style # rename it just for generality of running
```

#### Build

```bash
sudo docker built -t multi_style . # in project directory, or use the pulled image
```

#### Run

```bash
sudo docker run \
--gpus all \
-v <path-to-the-movie-file>:/app/movie \
-v <your-desired-output-directory-for-the-movies>:/app/output/ multi_style
```

It will launch the `run.py` script and will access your video and GPU device. 
You'll need to have `nvidia-docker` installed on your machine and the desired video
to be translated in 8 different styles. The docker image will produce the styled
videos into the output folder.

### Demo videos

Live demo:

[![outside-scene-with-friends](https://img.youtube.com/vi/eyMuIuqwkio/0.jpg)](https://youtu.be/eyMuIuqwkio)

* the live demo can be tried via installing all dependencies to your machine since there is some issues with opencv using the webcam in docker which I didn't wank to solve (~20 hours)

* the provided `run.py` script opens up your webcam, you can quit with pressing `q` or change styles with the press of button `c`


#### Demo videos

[![outside-scene-with-friends](https://img.youtube.com/vi/Py-t08dwXF8/0.jpg)](https://www.youtube.com/watch?v=Py-t08dwXF8)


