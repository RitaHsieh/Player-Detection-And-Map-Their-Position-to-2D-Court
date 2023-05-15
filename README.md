# Player Detection and Map Their Position to 2D Court

## Description
The project detected the player by CenterNet and devided the court by SegNet to obtain a perspective matrix that can converte the court to a 2D view. The player would be mapped to the 2D view with that perspective matrix.

Together with the project, we create a tool for labeling the court, which you can see in the labeling file.

### Method
![The structure]()

### Result
[![Result](http://img.youtube.com/vi/LnUmOx-sZfg/0.jpg)](http://www.youtube.com/watch?v=LnUmOx-sZfg "The result of Player-Detection-And-Map-Their-Position-to-2D-Court")
The player positions can be detected without complete player overlap, and when the camera covers all four corners of the penalty area, the player positions can be accurately mapped to the 2D field view.

### Next Step of this Project
- Improve the accuracy of detection of the ball
- Segment the court more precisely by training the model better