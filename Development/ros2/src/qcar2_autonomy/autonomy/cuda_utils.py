import os

try:
    import torch
except Exception:
    torch = None


class CudaSelectionError(RuntimeError):
    pass


def _env_false(name):
    value = os.environ.get(name)
    if value is None:
        return False
    return value.strip().lower() in {"0", "false", "no", "off", "cpu"}


def _env_true(name):
    value = os.environ.get(name)
    if value is None:
        return False
    return value.strip().lower() in {"1", "true", "yes", "on", "gpu", "cuda"}


def _log(logger, level, message):
    if logger is None:
        return
    log_fn = getattr(logger, level, None) or getattr(logger, "warn", None)
    if log_fn is not None:
        log_fn(message)


def _cpu_or_raise(logger, context, reason, allow_cpu_fallback):
    if allow_cpu_fallback:
        _log(logger, "warn", f"{context}: {reason}; using CPU")
        return "cpu"
    message = f"{context}: {reason}; CUDA is required and CPU fallback is disabled"
    _log(logger, "error", message)
    raise CudaSelectionError(message)


def select_yolo_device(
    logger=None,
    requested_device=0,
    use_cuda=True,
    context="YOLO",
    allow_cpu_fallback=True,
):
    """Return an Ultralytics device value, preferring CUDA and optionally failing closed."""
    if _env_true("QCAR2_REQUIRE_CUDA"):
        allow_cpu_fallback = False
    elif os.environ.get("QCAR2_ALLOW_CPU_FALLBACK") is not None:
        allow_cpu_fallback = not _env_false("QCAR2_ALLOW_CPU_FALLBACK")

    if _env_false("QCAR2_USE_CUDA") or _env_false("QCAR2_CUDA"):
        use_cuda = False

    if os.environ.get("QCAR2_FORCE_CPU", "").strip().lower() in {"1", "true", "yes", "on"}:
        use_cuda = False

    try:
        if isinstance(requested_device, str):
            requested = requested_device.strip().lower()
            if requested in {"cpu", "none", "off", "false", "-1"}:
                use_cuda = False
                device_index = -1
            elif requested.startswith("cuda:"):
                device_index = int(requested.split(":", 1)[1])
            elif requested == "cuda":
                device_index = 0
            else:
                device_index = int(requested)
        else:
            device_index = int(requested_device)
    except Exception:
        return _cpu_or_raise(
            logger,
            context,
            f"invalid device={requested_device!r}",
            allow_cpu_fallback,
        )

    if device_index < 0:
        use_cuda = False

    if not use_cuda:
        _log(logger, "info", f"{context}: CUDA disabled; using CPU")
        return "cpu"

    if torch is None:
        return _cpu_or_raise(
            logger,
            context,
            "PyTorch is not importable",
            allow_cpu_fallback,
        )

    try:
        if not torch.cuda.is_available():
            return _cpu_or_raise(
                logger,
                context,
                "CUDA is not available",
                allow_cpu_fallback,
            )

        device_count = torch.cuda.device_count()
        if device_index >= device_count:
            return _cpu_or_raise(
                logger,
                context,
                f"requested CUDA device {device_index}, but only {device_count} device(s) exist",
                allow_cpu_fallback,
            )

        major, minor = torch.cuda.get_device_capability(device_index)
        arch = f"sm_{major}{minor}"
        supported = set(torch.cuda.get_arch_list())
        if supported and arch not in supported:
            name = torch.cuda.get_device_name(device_index)
            return _cpu_or_raise(
                logger,
                context,
                f"CUDA device {device_index} ({name}, {arch}) is not supported "
                f"by this PyTorch build ({sorted(supported)})",
                allow_cpu_fallback,
            )

        name = torch.cuda.get_device_name(device_index)
        _log(logger, "info", f"{context}: using CUDA device {device_index} ({name}, {arch})")
        return device_index
    except CudaSelectionError:
        raise
    except Exception as exc:
        return _cpu_or_raise(
            logger,
            context,
            f"CUDA check failed ({exc})",
            allow_cpu_fallback,
        )


def is_cuda_runtime_error(err):
    text = str(err).lower()
    return (
        "cuda" in text
        or "cudnn" in text
        or "cublas" in text
        or "no kernel image is available" in text
        or "not compatible with the current pytorch installation" in text
        or "nvrm" in text
        or "operation not supported" in text
    )


def clear_cuda_cache():
    if torch is None:
        return
    try:
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        pass
