# dataClay and PyTorch --an opinionated and probably inaccurate quickstart

Let's start with a disclaimer: I am not really a PyTorch expert (or a ML expert or anything like that).
So if there are things that do not make sense, probably I am wrong. However, I did my best by following
the example on [Vision Transformer](https://github.com/pytorch/examples/blob/main/vision_transformer/).

## Start

### Preparing the environment

I recommend to work in a virtual environment. Note that the `requirements.txt` is intended to be used both
by the Docker image build and the Jupyter Notebook. Simply prepare a virtual environment, e.g.:

```bash
$ python3 -m venv venv
$ source venv/bin/activate
$ pip install -r requirements.txt
```

### Starting dataClay

A `docker-compose.yml` is provided for convenience (it is a overly simplified deployment):

```bash
$ docker compose up
```

Feel free to use `docker-compose` instead of `docker compose`, or add `-d` flag to have it on the
background, or use whatever flow suits your tastes.

### Training a sample model

Assuming that you have activated the virtual environment, just open Jupyter Notebook and open the
`Train.ipynb`. It shows the steps to connect to dataClay (the port is the default and opened by
docker-compose) and prepare a sample `torch` Neural Network.

## Development cycle

Typically, you will change stuff on the `model` folder. This means that you need to restart the
Jupyter Notebook kernel (to force re-import of modules). dataClay backend also needs to be restarted,
and you can do that with Docker as follows:

```bash
$ docker compose restart dataclay-backend
```
