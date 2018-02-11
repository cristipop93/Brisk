import org.opencv.calib3d.Calib3d;
import org.opencv.core.*;
import org.opencv.features2d.DescriptorExtractor;
import org.opencv.features2d.DescriptorMatcher;
import org.opencv.features2d.FeatureDetector;
import org.opencv.features2d.Features2d;
import org.opencv.imgcodecs.Imgcodecs;

import java.util.ArrayList;
import java.util.Iterator;
import java.util.LinkedList;
import java.util.List;

public class Main {
  static{
    System.loadLibrary(Core.NATIVE_LIBRARY_NAME);
  }

  public static void main(String[] args) {
    // Read the images
    Mat firstImageSrcImgMat = Imgcodecs.imread("image1.jpg");
    Mat secondImageSrcImgMat = Imgcodecs.imread("image2.jpg");

    if (firstImageSrcImgMat.empty() || secondImageSrcImgMat.empty()) {
      System.out.println("Failed to load images");
      return;
    }

    FeatureDetector featureDetector = FeatureDetector.create(FeatureDetector.BRISK);

    MatOfKeyPoint firstImgMatOfKeyPoints = new MatOfKeyPoint();
    MatOfKeyPoint secondImgMatOfKeyPoints = new MatOfKeyPoint();

    // 1. Using Brisk detect image for keyPoints
    featureDetector.detect(firstImageSrcImgMat, firstImgMatOfKeyPoints);
    featureDetector.detect(secondImageSrcImgMat, secondImgMatOfKeyPoints);

    System.out.println("Detected " + firstImgMatOfKeyPoints.size() + " and " + secondImgMatOfKeyPoints + " blobs in the images");

    List<KeyPoint> firstImgKeyPoints = firstImgMatOfKeyPoints.toList();
    List<KeyPoint> secondImgKeyPoints = secondImgMatOfKeyPoints.toList();

    System.out.println("First Image key points: " + firstImgKeyPoints);
    System.out.println("Second Image key points: " + secondImgKeyPoints);

    Mat firstImgDescriptors = new Mat();
    Mat secondImgDescriptors = new Mat();

    // 2. Build The Descriptor
    DescriptorExtractor extractor = DescriptorExtractor.create(DescriptorExtractor.BRISK);
    extractor.compute(firstImageSrcImgMat, firstImgMatOfKeyPoints, firstImgDescriptors);
    extractor.compute(secondImageSrcImgMat, secondImgMatOfKeyPoints, secondImgDescriptors);

    System.out.println("descriptorsA.size() : " + firstImgDescriptors.size());
    System.out.println("descriptorsB.size() : " + secondImgDescriptors.size());

    List<MatOfDMatch> matches = new ArrayList<>();

    // 3. Using Hamming distance determine the match
    DescriptorMatcher matcher = DescriptorMatcher.create(DescriptorMatcher.BRUTEFORCE_HAMMING); // BRUTEFORCE_HAMMING
    matcher.knnMatch(firstImgDescriptors, secondImgDescriptors, matches, 100);




    LinkedList<DMatch> good_matches = new LinkedList<DMatch>();
    for (Iterator<MatOfDMatch> iterator = matches.iterator(); iterator.hasNext();) {
      MatOfDMatch matOfDMatch = (MatOfDMatch) iterator.next();
      if (matOfDMatch.toArray()[0].distance / matOfDMatch.toArray()[1].distance < 0.9) {
        good_matches.add(matOfDMatch.toArray()[0]);
      }
    }

    // get keypoint coordinates of good matches to find homography and remove outliers using ransac
    List<Point> pts1 = new ArrayList<Point>();
    List<Point> pts2 = new ArrayList<Point>();
    for(int i = 0; i<good_matches.size(); i++){
      pts1.add(firstImgMatOfKeyPoints.toList().get(good_matches.get(i).queryIdx).pt);
      pts2.add(secondImgMatOfKeyPoints.toList().get(good_matches.get(i).trainIdx).pt);
    }

    // convertion of data types - there is maybe a more beautiful way
    Mat outputMask = new Mat();
    MatOfPoint2f pts1Mat = new MatOfPoint2f();
    pts1Mat.fromList(pts1);
    MatOfPoint2f pts2Mat = new MatOfPoint2f();
    pts2Mat.fromList(pts2);

    Mat Homog = Calib3d.findHomography(pts1Mat, pts2Mat, Calib3d.RANSAC, 15, outputMask, 2000, 0.995);

    // output contains zeros and ones indicating which matches are filtered
    LinkedList<DMatch> better_matches = new LinkedList<DMatch>();
    for (int i = 0; i < good_matches.size(); i++) {
      if (outputMask.get(i, 0)[0] != 0.0) {
        better_matches.add(good_matches.get(i));
      }
    }

    // DRAWING OUTPUT
    Mat outputImg = new Mat();
    // this will draw all matches
    MatOfDMatch better_matches_mat = new MatOfDMatch();
    better_matches_mat.fromList(better_matches);
    Features2d.drawMatches(firstImageSrcImgMat, firstImgMatOfKeyPoints, secondImageSrcImgMat, secondImgMatOfKeyPoints, better_matches_mat, outputImg);

    // save image
    Imgcodecs.imwrite("result.jpg", outputImg);

  }
}
