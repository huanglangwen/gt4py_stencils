docker build -t microph .
docker stop microph 1>/dev/null 2>/dev/null
docker rm microph 1>/dev/null 2>/dev/null
docker run -i -t --rm \
    --mount type=bind,source=`pwd`/../data_microph,target=/microph/data \
    --name=microph microph
