FROM python:3.8-slim

RUN groupadd -r user && useradd -m --no-log-init -r -g user user

RUN mkdir -p /opt/app /input /output \
    && chown user:user /opt/app /input /output

USER user
WORKDIR /opt/app

ENV PATH="/home/user/.local/bin:${PATH}"

RUN python -m pip install --user -U pip && python -m pip install --user pip-tools



COPY --chown=user:user requirements.txt /opt/app/
RUN python -m piptools sync requirements.txt

COPY --chown=user:user outcomes-model-11254.pt /opt/app/
COPY --chown=user:user outcomes-model-19529.pt /opt/app/
COPY --chown=user:user outcomes-model-660969.pt /opt/app/
COPY --chown=user:user outcomes-model-682359.pt /opt/app/
COPY --chown=user:user outcomes-model-944415.pt /opt/app/
COPY --chown=user:user outcomes-model-2150913.pt /opt/app/
COPY --chown=user:user outcomes-model-47554755.pt /opt/app/
COPY --chown=user:user outcomes-model-76607660.pt /opt/app/
COPY --chown=user:user outcomes-model-94835948.pt /opt/app/
COPY --chown=user:user outcomes-model-4618746.pt /opt/app/

COPY --chown=user:user utils.py /opt/app/
COPY --chown=user:user process.py /opt/app/

ENTRYPOINT [ "python", "-m", "process" ]
