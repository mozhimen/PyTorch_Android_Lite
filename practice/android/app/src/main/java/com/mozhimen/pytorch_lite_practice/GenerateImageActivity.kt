package com.mozhimen.pytorch_lite_practice

import android.graphics.Bitmap
import android.graphics.Color
import android.os.Bundle
import android.util.Log
import android.view.View
import android.widget.Button
import android.widget.ImageView
import android.widget.TextView
import androidx.appcompat.app.AppCompatActivity
import org.pytorch.IValue
import org.pytorch.LiteModuleLoader
import org.pytorch.Module
import org.pytorch.Tensor
import java.io.File
import java.io.FileOutputStream
import java.io.IOException
import java.util.Random
import kotlin.math.max
import kotlin.math.min


class GenerateImageActivity : AppCompatActivity() {
    // Elements in the view
    private var btnGenerate: Button? = null
    private var ivImage: ImageView? = null
    private var tvWaiting: TextView? = null

    // Tag used for logging
    private val TAG = "MainActivity2"

    // PyTorch model
    private var module: Module? = null

    // Size of the input tensor
    private var inSize = 512

    // Width and height of the output image
    private var width = 256
    private var height = 256

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_generate_image)

        // Get the elements in the activity
        btnGenerate = findViewById(R.id.btnGenerate)
        ivImage = findViewById(R.id.ivImage)
        tvWaiting = findViewById(R.id.tvWaiting)

        // Load in the model
        try {
            module = LiteModuleLoader.load(assetFilePath("imageGen.pt"))
        } catch (e: IOException) {
            Log.e(TAG, "Unable to load model", e)
        }

        // When the button is clicked, generate a new image
        // When the button is clicked, generate a new image
        btnGenerate!!.setOnClickListener {
            // Error handing
            btnGenerate!!.isClickable = false
            ivImage!!.visibility = View.INVISIBLE
            tvWaiting!!.visibility = View.VISIBLE

            // Prepare the input tensor. This time, its a
            // a single integer value.
            val inputTensor = generateTensor(inSize)

            // Run the process on a background thread
            Thread {
                // Get the output from the model. The
                // length should be 256*256*3 or 196608
                // Note that the output is in the layout
                // [R, G, B, R, G, B, ..., B] and we
                // have to deal with that.
                var outputArr =
                    module!!.forward(IValue.from(inputTensor)).toTensor().dataAsFloatArray

                // Ensure the output array has values between 0 and 255
                for (i in outputArr!!.indices) {
                    outputArr[i] = min(max(outputArr[i], 0f), 255f)
                }

                // Create a RGB bitmap of the correct shape
                var bmp = Bitmap.createBitmap(
                    width,
                    height,
                    Bitmap.Config.RGB_565
                )

                // Iterate over all values in the output tensor
                // and put them into the bitmap
                var loc = 0
                for (y in 0 until width) {
                    for (x in 0 until height) {
                        bmp.setPixel(
                            x,
                            y,
                            Color.rgb(
                                outputArr[loc].toInt(),
                                outputArr[loc + 1].toInt(),
                                outputArr[loc + 2].toInt()
                            )
                        )
                        loc += 3
                    }
                }

                // The output of the network is no longer needed
                outputArr = null

                // Resize the bitmap to a larger image
                bmp = Bitmap.createScaledBitmap(
                    bmp, 512, 512, false
                )

                // Display the image
                val finalBmp = bmp
                runOnUiThread {
                    ivImage!!.setImageBitmap(finalBmp)

                    // Error handing
                    btnGenerate!!.isClickable = true
                    tvWaiting!!.visibility = View.INVISIBLE
                    ivImage!!.visibility = View.VISIBLE
                }
            }.start()
        }
    }

    // Generate a tensor of random doubles given the size of
    // the tensor to generate
    private fun generateTensor(size: Int): Tensor? {
        // Create a random array of doubles
        val rand = Random()
        val arr = DoubleArray(size)
        for (i in 0 until size) {
            arr[i] = rand.nextGaussian()
        }

        // Create the tensor and return it
        val s = longArrayOf(1, size.toLong())
        return Tensor.fromBlob(arr, s)
    }

    // Given the name of the pytorch model, get the path for that model
    @Throws(IOException::class)
    fun assetFilePath(assetName: String): String? {
        val file = File(this.filesDir, assetName)
        if (file.exists() && file.length() > 0) {
            return file.absolutePath
        }
        this.assets.open(assetName).use { `is` ->
            FileOutputStream(file).use { os ->
                val buffer = ByteArray(4 * 1024)
                var read: Int
                while (`is`.read(buffer).also { read = it } != -1) {
                    os.write(buffer, 0, read)
                }
                os.flush()
            }
            return file.absolutePath
        }
    }
}