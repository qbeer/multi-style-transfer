ffmpeg -i ./movie /app/frames/frame_%04d.png

conda run -n pytorch python create_styled_movies.py

ffmpeg -framerate 25 -i /app/frames/frame_%04d_styled_0.png -c:v libx264 -r 30 -pix_fmt yuv420p /app/output/out0.mp4
ffmpeg -framerate 25 -i /app/frames/frame_%04d_styled_1.png -c:v libx264 -r 30 -pix_fmt yuv420p /app/output/out1.mp4
ffmpeg -framerate 25 -i /app/frames/frame_%04d_styled_2.png -c:v libx264 -r 30 -pix_fmt yuv420p /app/output/out2.mp4
ffmpeg -framerate 25 -i /app/frames/frame_%04d_styled_3.png -c:v libx264 -r 30 -pix_fmt yuv420p /app/output/out3.mp4
ffmpeg -framerate 25 -i /app/frames/frame_%04d_styled_4.png -c:v libx264 -r 30 -pix_fmt yuv420p /app/output/out4.mp4
ffmpeg -framerate 25 -i /app/frames/frame_%04d_styled_5.png -c:v libx264 -r 30 -pix_fmt yuv420p /app/output/out5.mp4
ffmpeg -framerate 25 -i /app/frames/frame_%04d_styled_6.png -c:v libx264 -r 30 -pix_fmt yuv420p /app/output/out6.mp4
ffmpeg -framerate 25 -i /app/frames/frame_%04d_styled_7.png -c:v libx264 -r 30 -pix_fmt yuv420p /app/output/out7.mp4
