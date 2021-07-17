package com.example.ar2;

import androidx.annotation.Nullable;
import androidx.appcompat.app.AppCompatActivity;

import android.content.Context;
import android.content.Intent;
import android.content.pm.PackageManager;
import android.graphics.Bitmap;
import android.graphics.Camera;
import android.net.Uri;
import android.os.Bundle;
import android.provider.MediaStore;
import android.util.Log;
import android.view.View;
import android.widget.Button;
import android.widget.ImageView;
import android.widget.TextView;

import com.example.ar2.ml.LiteModelMobilenetV3Small100224;

import org.tensorflow.lite.DataType;
import org.tensorflow.lite.Interpreter;
import org.tensorflow.lite.support.common.FileUtil;
import org.tensorflow.lite.support.common.ops.NormalizeOp;
import org.tensorflow.lite.support.image.ImageProcessor;
import org.tensorflow.lite.support.image.TensorImage;
import org.tensorflow.lite.support.image.ops.ResizeOp;
import org.tensorflow.lite.support.image.ops.ResizeWithCropOrPadOp;
import org.tensorflow.lite.support.image.ops.Rot90Op;
import org.tensorflow.lite.support.tensorbuffer.TensorBuffer;
import org.tensorflow.lite.support.model.Model;

import java.io.IOException;
import java.nio.ByteBuffer;
import java.nio.MappedByteBuffer;


public class MainActivity extends AppCompatActivity {

    private ImageView mimageview;
    private int ss=1;
    private String[] labels={"apple", "banana","beetroot", "bell pepper", "cabbage", "capsicum", "carrot", "cauliflower", "chilli pepper", "corn", "cucumber", "eggplant", "garlic", "ginger", "grapes", "jalepeno", "kiwi", "lemon", "lettuce", "mango", "onion", "orange", "paprika", "pear", "peas", "pineapple", "pomegranate", "potato", "raddish", "soy beans", "spinach", "sweetcorn", "sweetpotato", "tomato", "turnip", "watermelon"};
    private TextView tv;
    private static final int REQUEST_IMAGE_CAPTURE=101;
    private Button btn_select;
    private TensorImage inputImageBuffer;

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);
        mimageview=findViewById(R.id.ImageView);
        tv=findViewById(R.id.tv);
        btn_select=(Button) findViewById(R.id.btnCamera2);
        btn_select.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View view) {
                Intent intent=new Intent(Intent.ACTION_GET_CONTENT);
                intent.setType("image/*");
                startActivityForResult(intent,100);
            }
        });

    }

    @Override
    protected void onActivityResult(int requestCode, int resultCode, @Nullable @org.jetbrains.annotations.Nullable Intent data) {
        super.onActivityResult(requestCode, resultCode, data);
        if (requestCode==100)
        {
            mimageview.setImageURI(data.getData());

            Uri uri=data.getData();
            try {
                Bitmap img2=MediaStore.Images.Media.getBitmap(this.getContentResolver(),uri);
                Bitmap img=Bitmap.createScaledBitmap(img2,224,224,true);
                inputImageBuffer = new TensorImage();
                inputImageBuffer.load(img2);

                int cropSize = Math.min(img2.getWidth(), img2.getHeight());
                int numRoration = 0 / 90;
                ImageProcessor imageProcessor =
                        new ImageProcessor.Builder()
                                .add(new ResizeWithCropOrPadOp(cropSize, cropSize))
                                .add(new ResizeOp(224, 224, ResizeOp.ResizeMethod.NEAREST_NEIGHBOR))
                                .add(new Rot90Op(numRoration))
                                .add(new NormalizeOp(0f, 255f))
                                .build();
                TensorImage test=imageProcessor.process(inputImageBuffer);
                try {
                    LiteModelMobilenetV3Small100224 model = LiteModelMobilenetV3Small100224.newInstance(getApplicationContext());

                    // Creates inputs for reference.
                    TensorBuffer inputFeature0 = TensorBuffer.createFixedSize(new int[]{1, 224, 224, 3}, DataType.FLOAT32);

                    TensorImage tensorImage=new TensorImage(DataType.FLOAT32);
                    tensorImage.load(img);
//                    TensorImage tensorImage=TensorImage.fromBitmap(img);
                    ByteBuffer byteBuffer = test.getBuffer();

                    inputFeature0.loadBuffer(byteBuffer);

                    // Runs model inference and gets result.
                    LiteModelMobilenetV3Small100224.Outputs outputs = model.process(inputFeature0);
                    TensorBuffer outputFeature0 = outputs.getOutputFeature0AsTensorBuffer();

                    // Releases model resources if no longer used.
                    model.close();
                    int vtmax=0;
                    for (int i=0;i<outputFeature0.getFloatArray().length;i++)
                        if (outputFeature0.getFloatArray()[i]>outputFeature0.getFloatArray()[vtmax]) vtmax=i;
                    tv.setText(labels[vtmax]);
                } catch (IOException e) {
                    // TODO Handle the exception
                }
            } catch (IOException e) {
                e.printStackTrace();
            }
        } else
        {
            if (requestCode == REQUEST_IMAGE_CAPTURE && resultCode == RESULT_OK) {
            Bundle extras = data.getExtras();
            Bitmap img2=(Bitmap) extras.get("data");
            inputImageBuffer = new TensorImage();
            inputImageBuffer.load(img2.copy(Bitmap.Config.ARGB_8888,true));

            int cropSize = Math.min(img2.getWidth(), img2.getHeight());
            int numRoration = 0 / 90;
            ImageProcessor imageProcessor =
                    new ImageProcessor.Builder()
                            .add(new ResizeWithCropOrPadOp(cropSize, cropSize))
                            .add(new ResizeOp(224, 224, ResizeOp.ResizeMethod.NEAREST_NEIGHBOR))
                            .add(new Rot90Op(numRoration))
                            .add(new NormalizeOp(0f, 255f))
                            .build();
            TensorImage test=imageProcessor.process(inputImageBuffer);
            try {
                LiteModelMobilenetV3Small100224 model = LiteModelMobilenetV3Small100224.newInstance(getApplicationContext());

                // Creates inputs for reference.
                TensorBuffer inputFeature0 = TensorBuffer.createFixedSize(new int[]{1, 224, 224, 3}, DataType.FLOAT32);

                ByteBuffer byteBuffer = test.getBuffer();

                inputFeature0.loadBuffer(byteBuffer);

                // Runs model inference and gets result.
                LiteModelMobilenetV3Small100224.Outputs outputs = model.process(inputFeature0);
                TensorBuffer outputFeature0 = outputs.getOutputFeature0AsTensorBuffer();

                // Releases model resources if no longer used.
                model.close();
                int vtmax=0;
                for (int i=0;i<outputFeature0.getFloatArray().length;i++)
                    if (outputFeature0.getFloatArray()[i]>outputFeature0.getFloatArray()[vtmax]) vtmax=i;
                tv.setText(labels[vtmax]);
            } catch (IOException e) {
                // TODO Handle the exception
            }
            mimageview.setImageBitmap(img2.copy(Bitmap.Config.ARGB_8888,true));
        }
        }
    }

    public void Select(View view)
    {

    }
    public void takePicture(View view)
    {
        Intent imageTakeIntent=new Intent(MediaStore.ACTION_IMAGE_CAPTURE);
        if (imageTakeIntent.resolveActivity(getPackageManager())!=null)
        {
            startActivityForResult(imageTakeIntent,REQUEST_IMAGE_CAPTURE);
        }
    }

    public void KQ(View view)
    {

    }

//    @Override
//    protected void onActivityResult(int requestCode, int resultCode, @Nullable @org.jetbrains.annotations.Nullable Intent data) {
//        super.onActivityResult(requestCode, resultCode, data);
//        if (requestCode == REQUEST_IMAGE_CAPTURE && resultCode == RESULT_OK) {
//            Bundle extras = data.getExtras();
//            Bitmap imageBitmap = (Bitmap) extras.get("data");
//            Bitmap img1=Bitmap.createScaledBitmap(imageBitmap,224,224,true);
//            Bitmap img = img1.copy (Bitmap.Config.ARGB_8888, true);
////            try {
////                LiteModelMobilenetV3Small100224V1 model = LiteModelMobilenetV3Small100224V1.newInstance(getApplicationContext());
////
////                // Creates inputs for reference.
////                TensorBuffer inputFeature0 = TensorBuffer.createFixedSize(new int[]{1, 224, 224, 3}, DataType.FLOAT32);
////
////                TensorImage tensorImage=new TensorImage(DataType.FLOAT32);
////                tensorImage.load(img);
////                ByteBuffer byteBuffer = tensorImage.getBuffer();
////
////                inputFeature0.loadBuffer(byteBuffer);
////
////                // Runs model inference and gets result.
////                LiteModelMobilenetV3Small100224V1.Outputs outputs = model.process(inputFeature0);
////                TensorBuffer outputFeature0 = outputs.getOutputFeature0AsTensorBuffer();
////
////
////                // Releases model resources if no longer used.
////                model.close();
////            } catch (IOException e) {
////                // TODO Handle the exception
////            }
//
//            try {
//                LiteModelMobilenetV3Small100224 model = LiteModelMobilenetV3Small100224.newInstance(getApplicationContext());
//
//                // Creates inputs for reference.
//                TensorBuffer inputFeature0 = TensorBuffer.createFixedSize(new int[]{1, 224, 224, 3}, DataType.FLOAT32);
//
//                TensorImage tensorImage=new TensorImage(DataType.FLOAT32);
//                tensorImage.load(img);
//                ByteBuffer byteBuffer = tensorImage.getBuffer();
//
//                inputFeature0.loadBuffer(byteBuffer);
//
//                // Runs model inference and gets result.
//                LiteModelMobilenetV3Small100224.Outputs outputs = model.process(inputFeature0);
//                TensorBuffer outputFeature0 = outputs.getOutputFeature0AsTensorBuffer();
//
//                // Releases model resources if no longer used.
//                model.close();
//                int vtmax=0;
//                for (int i=0;i<outputFeature0.getFloatArray().length;i++)
//                    if (outputFeature0.getFloatArray()[i]>outputFeature0.getFloatArray()[vtmax]) vtmax=i;
//                tv.setText(labels[vtmax]+" "+outputFeature0.getFloatArray().length);
//            } catch (IOException e) {
//                // TODO Handle the exception
//            }
//            mimageview.setImageBitmap(imageBitmap);
//        }
//    }

}