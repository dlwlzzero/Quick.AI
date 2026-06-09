// SPDX-License-Identifier: Apache-2.0
/*
 * Copyright (C) 2026 Samsung Electronics Co., Ltd. All Rights Reserved.
 *
 * @file    VisionImageProcessor.kt
 * @brief   Shared vision image-processor interface and model-input type.
 *
 * Both [LlavaNextImageProcessor] and [SiglipImageProcessor] implement
 * [VisionImageProcessor] so the multimodal consumer in NativeQuickDotAI can be
 * typed against a single interface and select the concrete processor per model.
 */
package com.example.quickdotai

import android.graphics.Bitmap

/** Common surface for per-model image preprocessing. */
interface VisionImageProcessor {
    fun preprocess(image: Bitmap): VisionModelInput
    fun getCropSize(): Int
}

/**
 * The final model input for a single image.
 *
 * @param pixelValues Flattened pixel buffer ready for the native vision tower.
 * @param originalSize Original image size as Pair(height, width).
 */
data class VisionModelInput(val pixelValues: FloatArray, val originalSize: Pair<Int, Int>)
