function picture = imagePrediction(predictionImagePath, predictionImageName, format, trainedNetwork)
    % Tamaño del marco de entrada
    frameX = 7;
    frameY = 7;

    % Carga la imagen de entrada
    inputPicture = imread([predictionImagePath, predictionImageName, format]);

    % Ignora el marco circundante debido a la entrada incompleta de la red
    x = (frameX - 1) / 2;
    y = (frameY - 1) / 2;

    % Inicializa matrices
    currentImage = zeros(frameY, frameX);
    imageArray = zeros(frameX * frameY, (size(inputPicture, 1) - 2 * y) * (size(inputPicture, 2) - 2 * x));
    imageArrayOriginal = zeros(frameX * frameY, (size(inputPicture, 1) - 2 * y) * (size(inputPicture, 2) - 2 * x));

    % Itera a través de los píxeles de la imagen sin el marco circundante
    for i = 1 + y : size(inputPicture, 1) - y
        for j = 1 + x : size(inputPicture, 2) - x
            % Lee el marco
            currentImage = inputPicture(i - y : i + y, j - x : j + x, 1);
            % Reorganiza el marco de una matriz 7x7 a un vector columna 49x1
            currentImage = reshape(currentImage', [], 1);
            % Escribe el vector columna en la matriz
            imageArray(:, (i - (1 + x)) * (size(inputPicture, 2) - 2 * x) + (j - x)) = currentImage;
        end
    end

    % Inicializa la imagen de salida
    picture = zeros(size(inputPicture, 1), size(inputPicture, 2), 3);

    imageArrayOriginal = imageArray;
    imageArray = imageArray';

    % Prepara los datos para el entrenamiento
    imageArray = cast(imageArray, 'double');

    % Escala la entrada de [0;255] a [0;1] debido a la función sigmoide
    % Solo para valores de entrada entre [-4;4] la función sigmoide muestra diferencias significativas en la salida
    imageArray = imageArray / 255;

    % Prueba la red entrenada con la imagen
    labelArrayTest = networkPrediction(imageArray, trainedNetwork);
    labelArrayTest = round(labelArrayTest);

    % Reconstruye la imagen con la predicción de prueba
    for i = 1 + y : size(inputPicture, 1) - y
        for j = 1 + x : size(inputPicture, 2) - x
            prediction = labelArrayTest(:, (i - (1 + x)) * (size(inputPicture, 2) - 2 * x) + (j - x))';
            if isequal(prediction, [0, 0, 0, 0])
                picture(i, j, :) = imageArrayOriginal(25, (i - (1 + x)) * (size(inputPicture, 2) - 2 * x) + (j - x));
            elseif isequal(prediction, [1, 0, 0, 0])
                picture(i, j, :) = [255, 255, 64]; % amarillo oscuro
            elseif isequal(prediction, [0, 1, 0, 0])
                picture(i, j, :) = [0, 255, 0]; % verde
            elseif isequal(prediction, [0, 0, 1, 0])
                picture(i, j, :) = [0, 0, 255]; % azul
            elseif isequal(prediction, [0, 0, 0, 1])
                picture(i, j, :) = [255, 0, 0]; % rojo
            else
                picture(i, j, :) = imageArrayOriginal(25, (i - (1 + x)) * (size(inputPicture, 2) - 2 * x) + (j - x));
            end
        end
    end

    picture = cast(picture, 'uint8');
end
