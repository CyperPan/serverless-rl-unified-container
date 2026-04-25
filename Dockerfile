# Unified actor + learner image — Lambda-compatible.
#
# Same base image as the original sim-nitro/aws_lambda actor. Adds the
# learner module so the Lambda runtime (or any CMD-based runner) can
# dispatch by event['role'] without rebuilding.

FROM public.ecr.aws/lambda/python:3.10

# ── System deps shared by both roles ──────────────────────────────────
RUN yum -y install \
    mesa-libOSMesa-devel.x86_64 \
    mesa-libGL-devel.x86_64 \
    patchelf \
    gcc \
    && yum clean all

# ── Python deps (union of actor + learner; actor's set was already a superset) ──
COPY requirements.txt ${LAMBDA_TASK_ROOT}
RUN pip install --no-cache-dir -r ${LAMBDA_TASK_ROOT}/requirements.txt \
    && rm ${LAMBDA_TASK_ROOT}/requirements.txt

# ── MuJoCo native libs (kept for Hopper-v3 / mujoco-py path) ─────────
ARG LAMBDA_USER_PATH="/home/sbx_user1051"
ENV MUJOCO_PY_MUJOCO_PATH=$LAMBDA_USER_PATH/.mujoco/mujoco210
RUN mkdir -p $LAMBDA_USER_PATH/.mujoco/mujoco210
COPY mujoco210 $LAMBDA_USER_PATH/.mujoco/mujoco210
ENV LD_LIBRARY_PATH=${LD_LIBRARY_PATH}:$LAMBDA_USER_PATH/.mujoco/mujoco210/bin

# Newer Hopper-v4 uses the pip `mujoco` package (already in requirements.txt) —
# no extra system path needed.

# ── Source code ───────────────────────────────────────────────────────
COPY config.py             ${LAMBDA_TASK_ROOT}
COPY serverless_actor.py   ${LAMBDA_TASK_ROOT}
COPY serverless_learner.py ${LAMBDA_TASK_ROOT}
COPY handler.py            ${LAMBDA_TASK_ROOT}
COPY pre_compile.py        ${LAMBDA_TASK_ROOT}

# ── Build-time validation: walk both construction paths so the image
#    fails to build if either side is broken, and the image layer ships
#    with .pyc files for everything imported. ─────────────────────────
RUN python3 pre_compile.py

# ── Lambda entrypoint: handler dispatches role by event ──────────────
CMD [ "handler.handler" ]
