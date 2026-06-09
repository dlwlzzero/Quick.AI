// SPDX-License-Identifier: Apache-2.0
/*
 * Copyright (C) 2026 Samsung Electronics Co., Ltd. All Rights Reserved.
 *
 * @file    SiglipImageProcessor.kt
 * @brief   SigLIP image preprocessor for LFM2-VL (planar CHW FP32, 256x256).
 *
 * Bit-for-bit replica of the C++ reference in
 * nntrainer/Applications/CausalLM/image_util.h. The native SigLIP ViT consumes
 * exactly 3 * 256 * 256 = 196608 FP32 values in CHW (R plane, then G, then B),
 * normalized (v/255 - 0.5) / 0.5.
 *
 * Two steps:
 *   1. Simple bilinear resize to 256x256 on RGB uint8 (no half-pixel offset,
 *      round-half-away-from-zero via floor(v + 0.5f), clamp to [0,255]). If the
 *      source is already 256x256 the resize is skipped, matching the C++ path.
 *   2. CHW planar layout + normalization.
 *
 * Do NOT route this through PilloBilinearResizer — that is a different Pillow
 * fixed-point algorithm and would drift from the C++ reference.
 */
package com.example.quickdotai

import android.content.Context
import android.graphics.Bitmap
import android.graphics.Color
import kotlin.math.floor
import kotlin.math.min

/**
 * SigLIP preprocessor producing a single 256x256 planar-CHW FP32 tile.
 *
 * @param cropSize The square input size (256). Drives the consumer's
 *        numPatches = pixelValues.size / (cropSize*cropSize*3) arithmetic,
 *        which yields 1 for this single-tile layout.
 */
class SiglipImageProcessor(
    private val context: Context,
    private val cropSize: Int = 256,
) : VisionImageProcessor {

    override fun getCropSize(): Int = cropSize

    override fun preprocess(image: Bitmap): VisionModelInput {
        val srcW = image.width
        val srcH = image.height

        // Extract source RGB uint8 (HWC): src[(y*srcW + x)*3 + c]
        val pixels = IntArray(srcW * srcH)
        image.getPixels(pixels, 0, srcW, 0, 0, srcW, srcH)
        val src = ByteArray(srcW * srcH * 3)
        for (i in 0 until srcW * srcH) {
            val p = pixels[i]
            src[i * 3] = Color.red(p).toByte()
            src[i * 3 + 1] = Color.green(p).toByte()
            src[i * 3 + 2] = Color.blue(p).toByte()
        }

        // Step 1 — resize to 256x256 (skip if already exactly 256x256).
        val rgb: ByteArray = if (srcW == cropSize && srcH == cropSize) {
            src
        } else {
            resizeBilinear(src, srcW, srcH, cropSize, cropSize)
        }

        // Step 2 — CHW planar + normalize (v/255 - 0.5) / 0.5.
        val plane = cropSize * cropSize
        val output = FloatArray(3 * plane)
        for (c in 0 until 3) {
            for (y in 0 until cropSize) {
                for (x in 0 until cropSize) {
                    val v = (rgb[(y * cropSize + x) * 3 + c].toInt() and 0xFF)
                    output[c * plane + y * cropSize + x] = (v / 255f - 0.5f) / 0.5f
                }
            }
        }

        // originalSize convention matches LlavaNext: Pair(height, width).
        return VisionModelInput(pixelValues = output, originalSize = Pair(srcH, srcW))
    }

    /**
     * Simple bilinear resize on RGB uint8 (HWC). No half-pixel offset.
     * Rounds via floor(v + 0.5f) to match C++ std::round for non-negative
     * values, then clamps to [0,255].
     */
    private fun resizeBilinear(
        src: ByteArray,
        srcW: Int,
        srcH: Int,
        dstW: Int,
        dstH: Int,
    ): ByteArray {
        val dst = ByteArray(dstW * dstH * 3)
        val xRatio = srcW.toFloat() / dstW
        val yRatio = srcH.toFloat() / dstH
        for (y in 0 until dstH) {
            val py = y * yRatio
            val y0 = floor(py).toInt()
            val y1 = min(y0 + 1, srcH - 1)
            val fy = py - y0
            for (x in 0 until dstW) {
                val px = x * xRatio
                val x0 = floor(px).toInt()
                val x1 = min(x0 + 1, srcW - 1)
                val fx = px - x0
                for (c in 0 until 3) {
                    val v00 = (src[(y0 * srcW + x0) * 3 + c].toInt() and 0xFF).toFloat()
                    val v10 = (src[(y0 * srcW + x1) * 3 + c].toInt() and 0xFF).toFloat()
                    val v01 = (src[(y1 * srcW + x0) * 3 + c].toInt() and 0xFF).toFloat()
                    val v11 = (src[(y1 * srcW + x1) * 3 + c].toInt() and 0xFF).toFloat()
                    val v0 = v00 * (1 - fx) + v10 * fx
                    val v1 = v01 * (1 - fx) + v11 * fx
                    var out = floor((v0 * (1 - fy) + v1 * fy) + 0.5f).toInt()
                    if (out < 0) out = 0
                    if (out > 255) out = 255
                    dst[(y * dstW + x) * 3 + c] = out.toByte()
                }
            }
        }
        return dst
    }
}
