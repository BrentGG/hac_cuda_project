# HAC Cuda Project

## To Do:

- [x] 2D convolutie uitrekenen GPU
- [x] 2D convolutie uitrekenen CPU
- [x] min/max pooling berekenen GPU
- [x] min/max pooling berekenen CPU
- [x] afbeeldingen via URL inlezen CPU
- [x] afbeeldingen via URL inlezen GPU
- [x] Nsight compute analyse
- [x] Nsight systems analyse

## Gebruik

1. Open de [notebook](https://github.com/BrentGG/hac_cuda_project/blob/main/HAC_CUDA_project.ipynb) in Google Collab.
2. Run de eerste code block onder de hoofding 'Algemeen' om de git repo in te laden.
3. In zowel de ``opdracht_c.cu`` als de ``opdracht_cuda.cu`` kan de convolution kernel worden aangepast door de parameter aan te passen op respectievelijk lijn 251 en 245. By default is dit de kernel die edge detection doet.
4. Om de C versie te runnen, run beide code blocken onder de hoofding 'C Versie'.
5. Om de Cuda versie te runnen, run beide code blocken onder de hoofding 'Cuda Versie'.
6. De outputs van beide versies zijn 10 images genaamd ``image1.png`` t.e.m. ``image10.png`` en per image een ``imageX_convolution.png``, ``imageX_maxpool.png``, ``imageX_minpool.png`` en ``imageX_avgpool.png``.
7. De [Nsight compute analyse](https://github.com/BrentGG/hac_cuda_project/blob/main/opdracht_cuda.ncu-rep) bevindt zich in de repo. Om de Nsight compute analyse opnieuw te genereren, run de code blocken onder de hoofding 'Nsight Compute'.
8. De [Nsight systems analyse](https://github.com/BrentGG/hac_cuda_project/blob/main/report1.qdrep) bevindt zich in de repo. Om de Nsight systems analyse opnieuw te genereren, run de code blocken onder de hoofding 'Nsight Systems'.

## Resultaten

Het uitvoeren van enkel de convolutie en pooling functies in de C versie duurt ongeveer 700-800ms. In de Cuda versie is dit slechts 2-3ms. Echter het uitvoeren van het volledige programma van de C versie duurt in totaal ongeveer 5s 500ms. Voor de Cuda versie is dit ongeveer 5s 200ms. De Cuda versie is in totaal dus niet veel sneller dan de C versie, dit komt door de overhead van het alloceren van memory op de GPU en het feit dat we geen gebruik maken van streams. Als we grotere afbeeldingen zouden gebruiken zou het verschil wel een stuk groter zijn en zou dus het voordeel van het GPU gebruik duidelijker zijn.
