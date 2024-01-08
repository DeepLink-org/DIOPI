# Copyright (c) 2023, DeepLink.
# -*- coding: UTF-8 -*-
import numpy as np
from diopilib import build_generator_state
from .diopi_runtime import Tensor, Generator, default_context
from . import diopi_functions as F


class ManualTest(object):
    def test_dropout_(func, input, p=0.5, training=True, inplace=False):
        input_numpy = input.numpy()
        state = build_generator_state(input.context())
        generator = Generator(state)
        out, mask = func(input, p, training, inplace, generator)
        name = 'dropout' if func == F.dropout else 'dropout2d'
        out_numpy = out.numpy()
        mask_numpy = mask.numpy()

        rtol = 1e-2 if input_numpy.dtype == np.float16 else 1e-4
        atol = 5e-2 if input_numpy.dtype == np.float16 else 1e-5

        if training and input.numel() > 0:
            # compute ratio
            real_ratio = np.sum(mask_numpy) / mask.numel()
            # check data
            if func == F.dropout2d:
                tmp = np.ones(input.shape().data)
                mask_numpy = mask_numpy * tmp
            remains = out_numpy[mask_numpy == 1]
            ref = input_numpy[mask_numpy == 1]
            assert np.allclose(remains, ref / (1 - p), rtol=rtol, atol=atol), \
                f"failed to execute {name}, dropout value doesn't matches."
            if mask.numel() > 100:
                # 0.05 is from pytorch
                assert np.abs(real_ratio - (1 - p)) < 0.05, \
                    f"failed to execute {name}, dropout proportion unexpected."
        else:
            assert np.allclose(input_numpy, out_numpy, rtol=rtol, atol=atol), \
                f"failed to execute {name}, dropout value should be the same."

    def test_dropout(input, p=0.5, training=True, inplace=False):
        ManualTest.test_dropout_(F.dropout, input, p, training, inplace)

    def test_dropout2d(input, p=0.5, training=True, inplace=False):
        ManualTest.test_dropout_(F.dropout2d, input, p, training, inplace)

    def test_randperm(n):
        state = build_generator_state(default_context)
        generator = Generator(state)
        out = F.randperm(n, generator=generator)
        out_numpy = out.numpy()
        out_ref = np.arange(0, n, 1)
        if out.numel() > 10:
            assert not np.allclose(out_numpy, out_ref, 1e-3), \
                "failed to execute randperm"

        out_numpy.sort()
        assert np.allclose(out_numpy, out_ref, 1e-3), \
            "failed to execute randperm"

    def test_uniform(input, start=0, end=1, inplace=True):
        state = build_generator_state(input.context())
        generator = Generator(state)
        out = F.uniform(input, start, end, generator, inplace)
        epsilon = 1e-5   # eliminate minor precision error
        out_numpy = out.numpy()
        assert (out_numpy <= (end + epsilon)).all() and (out_numpy >= (start - epsilon)).all(), \
            "failed to execute uniform"
        if out.numel() > 100:
            assert abs(out_numpy.mean() - (end + start) / 2) < 1e-1, \
                "failed to execute uniform"

    def test_bernoulli(input, inplace=False, p=None):
        p_numpy = input.numpy()
        if input.numel() > 0:
            p = p_numpy.mean() if p is None else p
        state = build_generator_state(input.context())
        generator = Generator(state)
        input_origin = np.copy(input.numpy())
        out = F.bernoulli(input, inplace, p, generator)
        if inplace is False and p is None:
            assert np.allclose(input_origin, input.numpy()) is True, \
                "input changed"
        out_numpy = out.numpy()

        assert np.all((out_numpy == 0) | (out_numpy == 1)), "bernoulli output must be 0 or 1"
        if out.numel() > 100:
            assert abs(out_numpy.mean() - p) < 1e-1, \
                "failed to execute bernoulli"

    def test_random(input, start, end):
        state = build_generator_state(input.context())
        generator = Generator(state)
        out = F.random(input, start, end, generator)
        out_numpy = out.numpy()

        assert (out_numpy >= start).all(), \
            "failed to execute random"
        if end is not None:
            assert (out_numpy <= end - 1).all(), \
                "failed to execute random"

    def test_randn(size):
        from scipy import stats
        out = F.randn(size)
        out_numpy = out.numpy().flatten()
        p_value = stats.kstest(out_numpy, 'norm', args=(0.0, 1.))[1]
        # pytorch uses 0.0001
        assert p_value > 0.0001, f"can't pass the ks test, failed to execute normal, p_value is {p_value}"

    def test_normal(mean, std, size=None):
        from scipy import stats
        state = build_generator_state(default_context)
        generator = Generator(state)
        out = F.normal(mean, std, size, generator)
        out_numpy = out.numpy()
        if isinstance(mean, Tensor):
            mean_numpy = mean.numpy()
            out_numpy -= mean_numpy
            mean = 0.0
        if isinstance(std, Tensor):
            out_numpy -= mean
            std_numpy = std.numpy()
            out_numpy /= std_numpy
            mean = 0.0
            std = 1.
        out_numpy = out_numpy.flatten()
        if len(out_numpy) == 0:
            return True
        p_value = stats.kstest(out_numpy, 'norm', args=(mean, std + 1e-22))[1]
        assert p_value > 0.0001, f"can't pass the ks test, failed to execute normal, p_value is {p_value}"

    def test_normal_(input, mean, std, shape=None):
        from scipy import stats
        input_size = 0 in input.size().data
        state = build_generator_state(input.context())
        generator = Generator(state)
        out = F.normal_(input, mean, std, shape, generator)
        out_numpy = out.numpy()
        out_numpy = out_numpy.flatten()
        if len(out_numpy) == 0 and input_size:
            return True
        p_value = stats.kstest(out_numpy, 'norm', args=(mean, std))[1]
        assert p_value > 0.0001, f"can't pass the ks test, failed to execute normal_, p_value is {p_value}, shape of out is: {out_numpy.shape}"

    def test_multinomial(input, num_samples, replacement=False):
        state = build_generator_state(input.context())
        generator = Generator(state)
        out = F.multinomial(input, num_samples, replacement, generator)
        out_numpy = out.numpy()
        has_duplicates = False
        if 0 in input.size().data:
            assert len(out_numpy) == 0, "failed to execute multinomial"
        elif out.size().len == 2:
            has_duplicates = len(out_numpy[0]) != len(set(out_numpy[0]))
        else:
            has_duplicates = len(out_numpy) != len(set(out_numpy))
        if not replacement:
            assert has_duplicates is False, "failed to execute multinomial"
        out_numpy = out_numpy.flatten()
        assert len(out_numpy) % num_samples == 0, "failed to execute multinomial"
