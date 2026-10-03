# Preprocesamiento

Este analisis sera desarollado a base de la metologia CRISP-DM

## Seleccion de variables

Se a escogido como Target las variable **diagnosis** de tipo numerica, esta variable nos ayudaria a predecir a un paciente padezca de cancer.

Para las variables feature no se incluyeron las variables **Radius** y **Perimeter** , esto **debido a que la variable area tiene la informacion suficiente de la celula sobre el espacio que usa en las imagenes** , en otras palabras con esa parametro ya se sabe el espacio que ocupan las celulas malugnas en el organismo.

### Resultado del preprocesamiento 

El dataset quedo con una diferencia de 7 columnas de la original, quedo con un total de 25 columnas. Con mucha mejor variacion en el VIF. **El dataset no presento valores nulos, por ende no fue necesario tratamiento en ese apartado**