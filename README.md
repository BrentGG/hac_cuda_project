# hac_cuda_project

## To Do:

- [ ] 2D convolutie uitrekenen (zie deel 1) GPU
- [x] 2D convolutie uitrekenen (zie deel 1) CPU
- [ ] min/max pooling berekenen (zie deel 2) GPU
- [X] min/max pooling berekenen (zie deel 2) CPU
- [ ] afbeeldingen via URL inlezen CPU
- [ ] afbeeldingen via URL inlezen GPU (streams)

![titel opdracht](images/cudaTaakTitel.png?raw=true)

## Deel 1:
![opdracht 1](images/cudaTaakDeel1.png?raw=true)

## Deel 2:
![opdracht 2](images/cudaTaakDeel2.png?raw=true)

## getting started (WIP)

1.- open nieuw notebook in google colab [link](https://colab.research.google.com/drive/11K5aESAQQHsml9ied-BsuLnP-zG6wLMZ).<br/>

2.- geef onderstaande code in (naamgeving vrij te kiezen)<br/>
  %%writefile opdrachtHAC.cu <br/>
  "<insert code (opdrachtHAC.cu)>"  <br/>
  
3.- plak afbeelding en stb_image.h en stb_image_write.h bij bestanden <br/>

4.- geef onderstaande code in (zelfde naamgeving als 2de stap, "test" vrij te kiezen) <br/>
  !nvcc opdrachtHAC.cu -o test <br/>
  
5.- geef onderstaande code in (zelfde naamgeving als stap 4, afbeelding met correcte naamgeving) <br/>
  !./test mario.png <br/>
  
  
  output atm zou het een gray scale moeten zijn (begin punt opdracht)

