# Player Detection and Map Their Position to 2D Court

## Description
The project detected the player by CenterNet and devided the court by SegNet to obtain a perspective matrix that can converte the court to a 2D view. The player would be mapped to the 2D view with that perspective matrix.
![The structure](https://github.com/RitaHsieh/Player-Detection-And-Map-Their-Position-to-2D-Court/blob/main/explain_picture/structure.png)

Together with the project, we create a tool for labeling the court, which you can see in the labeling file.

### Result
Click to watch video:

[![Result](http://img.youtube.com/vi/LnUmOx-sZfg/0.jpg)](http://www.youtube.com/watch?v=LnUmOx-sZfg "The result of Player-Detection-And-Map-Their-Position-to-2D-Court")

The player positions can be detected without complete player overlap, and when the camera covers all four corners of the penalty area, the player positions can be accurately mapped to the 2D field view.

### Contribution
1. Create two datasets
    - Object tracking: 200 data sets of players, umpires and balls
    - Pitch segmentation: 965 pitch data sets
2. Create a court data set annotation tool
3. Combine DLA-34 into CenterNet
4. Using CenterNet (with hourglass) to obtain player coordinates, mAP reached 92.62
5. Use SegNet (Tversky loss) to obtain the pitch transition matrix
6. Converge CenterNet and Segnet results to map the player positions in the image to the plane coordinates

### Next Step of this Project
- Improve the accuracy of detection of the ball
- Segment the court more precisely by training the model better