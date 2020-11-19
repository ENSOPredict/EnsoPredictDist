# EnsoPredict

## Description

El Nino Southern Oscillation is a kind of climate phenomenon, which contains three states: El Nino, La Nina, and Neutral, and it requires certain changes in both the ocean and the atmosphere. El Nino indicates a warming ocean surface, while La Nina indicates a cooling ocean surface. To be specific, A cycle of ENSO is usually as long as 2-7 years, and would cause serious climate damage to some areas.

Former physical model uses the climate data and physical equations to describe laws of physics. Latitude and horizontal data is required to form a complete feature for ENSO detection, making the whole system computationally expensive.

In this project, we will develop regression and classification model to predict ONI, a rolling 3-month average of sea surface temperatures in the Nino 3.4 Region. We will also try CNN model to predict a similar result.




### Files:

- nino34index_ml_prediction.ipynb = Main notebook file to be open in jupyter notebook to run prediction

- nino34index_ml_prediction.py = Script called by nino34index_ml_prediction.ipynb to do prediction and visualization

- requirements.txt = Docker environment dependencies




### Folder: 

mount = folder to be mounted to docker

--data = child folder of mount to keep raw data

--model = child folder of mount to keep models with different lead times

--output = child folder of mount to keep output file of predictions




## Run Project using Docker

This instruction below is showing how to run EnsoPredict using docker in local computer. For Windows machine, docker can be executed by having Docker Desktop installed and running. 


a. Make sure you have docker service installed and running in your local computer. For example in this project, Docker Desktop was already running in Windows machine. Open windows PowerShell and start typing below commands.


b. Do git clone github repository by running: 

```
git clone https://github.com/ENSOPredict-Team2/EnsoPredictDist.git
```


c. Outside of powershell consoles, open file explorer and get into EnsoPredictDist\mount folder (according to each local machine path). Notice there are two folders inside it named "data" and "model" folder. This folder will hold the data and model files which can be downloaded from:

- Download all files below into "data" folder from: 

Nino 3.4 Index: https://doi.org/10.6084/m9.figshare.13227473.v1

Sea Surface Temperature: http://portal.nersc.gov/project/dasrepo/AGU_ML_Tutorial/sst.mon.mean.trefadj.anom.1880to2018.nc

regridded_era_t2m_anomalies.nc: https://doi.org/10.6084/m9.figshare.13232795.v1

- Download files below into "model" folder from: 

Lead Time 1 model: https://doi.org/10.6084/m9.figshare.13232771.v1

Lead Time 2 model: https://doi.org/10.6084/m9.figshare.13232783.v1

Lead Time 3 model: https://doi.org/10.6084/m9.figshare.13227608.v1 

Lead Time 4 model: https://doi.org/10.6084/m9.figshare.13227599.v1 

Lead Time 5 model: https://doi.org/10.6084/m9.figshare.13232789.v1

Lead Time 6 model: https://doi.org/10.6084/m9.figshare.13232795.v1

Lead Time 7 model: https://doi.org/10.6084/m9.figshare.13232795.v1

Lead Time 8 model: https://doi.org/10.6084/m9.figshare.13232795.v1

Lead Time 9 model: https://doi.org/10.6084/m9.figshare.13232795.v1

Lead Time 10 model: https://doi.org/10.6084/m9.figshare.13232795.v1

Lead Time 11 model: https://doi.org/10.6084/m9.figshare.13232795.v1

Lead Time 12 model: https://doi.org/10.6084/m9.figshare.13232795.v1


d. Back to powershell console, go into EnsoPredictDist folder:

```
cd EnsoPredictDist
```


e. Build docker image using Dockerfile with prefered image name, in this case "ensodocker"

```
docker build . -t ensodocker
```


f. Check that image was successfully created, run:
 
```
docker images -a
``` 

See that image ensodocker is among the result.


g. Run docker image in container by executing: (but need to change [path_to_mount_folder] with correct path first)

```
docker run --rm -it -d -p 8888:8888/tcp -v [path_to_mount_folder]:/src/mount ensodocker
```

for example if "mount" folder is located at 'D:\"DSCI 560"\EnsoPredictDist\mount' then the execution command will be:

```
docker run --rm -it -d -p 8888:8888/tcp -v D:\"DSCI 560"\EnsoPredictDist\mount:/src/mount ensodocker
```


h. Check container name running ensodocker and see the name under NAMES header:

```
docker ps
```


i. Execute the command below to run jupyter notebook for EnsoPredict: (but need to change [container_name] with correct name from step "h" above)

```
docker exec -it [container_name] jupyter notebook --port=8888 --no-browser --ip=0.0.0.0 --allow-root
```

for example if container name is "nostalgic_nash" then the execution command will be:

```
docker exec -it nostalgic_nash jupyter notebook --port=8888 --no-browser --ip=0.0.0.0 --allow-root
```


j. The powershell will show that jupyter notebook is running. Copy the link that being shown in the console and open in browser:

<img src="jupyter.jpg"/>


k. In Jupyter notebook browser, open or click nino34index_ml_prediction.ipynb file. Input parameters according to instruction in page and run predictions.


l. Quit jupyter notebook when finished by clicking Quit button at top right corner of jupyter tree browser.


m. Stop container by running: (but need to change [container_name] with correct name from step "h" above)

```
docker stop [container_name]
```

or using previous example container name then the command will be:

```
docker stop nostalgic_nash
```


n. Input or output files can still be accessible inside local "mount" folder even after docker container has been stopped.


### MISC:

a. To stop all running docker container, run:

```
docker stop $(docker ps -aq)
```

b. To delete all docker container, run:

```
docker rm -vf $(docker ps -aq)
```

c. To delete all docker images, run:

```
docker rmi -f $(docker images -aq)
```
