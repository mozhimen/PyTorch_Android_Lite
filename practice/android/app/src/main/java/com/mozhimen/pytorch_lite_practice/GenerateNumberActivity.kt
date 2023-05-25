package com.mozhimen.pytorch_lite_practice

import android.os.Bundle
import android.util.Log
import android.view.View
import android.widget.Button
import android.widget.EditText
import android.widget.TextView
import android.widget.Toast
import androidx.appcompat.app.AppCompatActivity
import org.pytorch.IValue
import org.pytorch.LiteModuleLoader
import org.pytorch.Module
import org.pytorch.Tensor
import java.io.File
import java.io.FileOutputStream
import java.io.IOException
import java.util.Random


class GenerateNumberActivity : AppCompatActivity() {
    // Elements in the view
    var etNumber: EditText? = null
    var btnInfer: Button? = null
    var tvDigits: TextView? = null

    // Tag used for logging
    private val TAG = "MainActivity"

    // PyTorch model
    var module: Module? = null

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_generate_number)

        // Get all the elements
        etNumber = findViewById(R.id.etNumber)
        btnInfer = findViewById(R.id.btnInfer)
        tvDigits = findViewById(R.id.tvDigits)

        // Load in the model
        try {
            module = LiteModuleLoader.load(assetFilePath("model.pt"))
        } catch (e: IOException) {
            Log.e(TAG, "Unable to load model", e)
        }

        // When the button is clicked, generate a noise tensor
        // and get the output from the model
        btnInfer!!.setOnClickListener(View.OnClickListener { // Error checking
            if (etNumber!!.text.toString().isEmpty()) {
                Toast.makeText(this@GenerateNumberActivity, "A number must be supplied", Toast.LENGTH_SHORT)
                    .show()
                return@OnClickListener
            }

            // Get the number of numbers to generate from the edit text
            val N = etNumber!!.text.toString().toInt()

            // More error checking
            if (N < 1 || N > 10) {
                Toast.makeText(
                    this@GenerateNumberActivity,
                    "Digits must be greater than 0 and less than 10",
                    Toast.LENGTH_SHORT
                ).show()
                return@OnClickListener
            }

            // Prepare the input tensor (N, 2)
            val shape = longArrayOf(N.toLong(), 2)
            val inputTensor = generateTensor(shape)

            // Get the output from the model
            val output = module!!.forward(IValue.from(inputTensor)).toTensor().dataAsLongArray

            // Get the output as a string
            var out = ""
            for (l in output) {
                out += l.toString()
            }

            // Show the output
            tvDigits!!.text = out
        })
    }

    // Generate a tensor of random numbers given the size of that tensor.
    fun generateTensor(Size: LongArray): Tensor? {
        // Create a random array of floats
        val rand = Random()
        val arr = FloatArray((Size[0] * Size[1]).toInt())
        for (i in 0 until Size[0] * Size[1]) {
            arr[i.toInt()] = -10000 + rand.nextFloat() * 20000
        }

        // Create the tensor and return it
        return Tensor.fromBlob(arr, Size)
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