# Optional Linux/CUDA runtime. Requires the NVIDIA Container Toolkit on the host.
FROM pytorch/pytorch:2.9.0-cuda12.6-cudnn9-runtime
WORKDIR /opt/FOCUS
ENV MPLBACKEND=Agg PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4
COPY . /opt/FOCUS
RUN python -m pip install --no-cache-dir .
CMD ["python", "demos/run.py"]
