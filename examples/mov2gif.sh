fn=`basename $1 .mov`
ffmpeg -i $fn.mov -vf "fps=10,scale=300:-1:flags=lanczos,split[s0][s1];
  [s0]palettegen=stats_mode=diff[p];[s1][p]paletteuse=dither=bayer" $fn.gif
