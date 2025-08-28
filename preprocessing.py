import logging
import os
import tempfile
from typing import List, Optional, Tuple

import cv2
import numpy as np
import torch
from PIL import Image

from config import IMAGE_DIR, PREPROCESSED_DIR, IMAGE_SIZE, MEAN, STD

logger = logging.getLogger(__name__)


class ImagePreprocessor:
    """
    Image preprocessing utilities for the computer vision project.

    ...

    Attributes
    ----------
    image_dir : str
        Path to the directory containing input images.
    preprocessed_dir : str
        Path to the directory where preprocessed images will be saved.
    image_size : Tuple[int, int]
        Size to which input images will be resized.
    mean : Tuple[float, float, float]
        Mean values for image normalization.
    std : Tuple[float, float, float]
        Standard deviation values for image normalization.

    Methods
    -------
    preprocess_images(self)
        Preprocesses all images in the input directory.
    resize_image(self, image, size)
        Resizes an image to the specified size.
    normalize_image(self, image)
        Normalizes an image using mean and standard deviation values.
    save_preprocessed_image(self, image, filename)
        Saves a preprocessed image to the output directory.
    """

    def __init__(
        self,
        image_dir: str = IMAGE_DIR,
        preprocessed_dir: str = PREPROCESSED_DIR,
        image_size: Tuple[int, int] = IMAGE_SIZE,
        mean: Tuple[float, float, float] = MEAN,
        std: Tuple[float, float, float] = STD,
    ):
        self.image_dir = image_dir
        self.preprocessed_dir = preprocessed_dir
        self.image_size = image_size
        self.mean = mean
        self.std = std

        if not os.path.exists(self.preprocessed_dir):
            os.makedirs(self.preprocessed_dir)

    def preprocess_images(self) -> None:
        """
        Preprocesses all images in the input directory.

        Resizes, normalizes, and saves images to the output directory.

        Raises
        ------
        FileNotFoundError
            If no images are found in the input directory.

        Returns
        -------
        None
        """
        logger.info("Starting image preprocessing...")

        if not os.listdir(self.image_dir):
            raise FileNotFoundError("No images found in the input directory.")

        for filename in os.listdir(self.image_dir):
            image = self.resize_image(Image.open(os.path.join(self.image_dir, filename)))
            normalized_image = self.normalize_image(image)
            self.save_preprocessed_image(normalized_image, filename)

        logger.info("Image preprocessing completed.")

    def resize_image(self, image: Image.Image) -> np.ndarray:
        """
        Resizes an image to the specified size.

        Parameters
        ----------
        image : PIL.Image.Image
            Input image to be resized.

        Returns
        -------
        np.ndarray
            Resized image as a numpy array.
        """
        return np.array(image.resize(self.image_size))

    def normalize_image(self, image: np.ndarray) -> torch.Tensor:
        """
        Normalizes an image using mean and standard deviation values.

        Parameters
        ----------
        image : np.ndarray
            Input image to be normalized.

        Returns
        -------
        torch.Tensor
            Normalized image as a torch tensor.
        """
        image = image.astype(np.float32) / 255.0
        image = (image - np.array(self.mean)) / np.array(self.std)
        return torch.from_numpy(image.transpose((2, 0, 1)))

    def save_preprocessed_image(
        self, image: torch.Tensor, filename: str
    ) -> Optional[str]:
        """
        Saves a preprocessed image to the output directory.

        Parameters
        ----------
        image : torch.Tensor
            Preprocessed image to be saved.
        filename : str
            Name of the file to save.

        Returns
        -------
        Optional[str]
            Path to the saved file, or None if an error occurred.
        """
        try:
            filename, file_extension = os.path.splitext(filename)
            tmp_file = tempfile.NamedTemporaryFile(
                suffix=file_extension, delete=False
            )
            path = tmp_file.name
            cv2.imwrite(path, (255 * image.numpy().transpose((1, 2, 0))).astype(np.uint8))
            os.replace(path, os.path.join(self.preprocessed_dir, filename + ".jpg"))
            return path
        except Exception as e:
            logger.error(f"Error saving preprocessed image: {e}")
            return None


def main():
    """
    Entry point for the image preprocessing script.

    Instantiates the ImagePreprocessor class and calls the preprocess_images method.

    Returns
    -------
    None
    """
    preprocessor = ImagePreprocessor()
    preprocessor.preprocess_images()


if __name__ == "__main__":
    main()


class ImagePreprocessorIntegrationTest:
    """
    Integration test suite for the ImagePreprocessor class.

    ...

    Attributes
    ----------
    preprocessor : ImagePreprocessor
        Instance of the ImagePreprocessor class.

    Methods
    -------
    test_preprocess_images(self)
        Tests the preprocess_images method.
    test_resize_image(self)
        Tests the resize_image method.
    test_normalize_image(self)
        Tests the normalize_image method.
    test_save_preprocessed_image(self)
        Tests the save_preprocessed_image method.

    """

    def setup_method(self) -> None:
        """
        Sets up the test environment.

        Instantiates the ImagePreprocessor class.

        Returns
        -------
        None
        """
        self.preprocessor = ImagePreprocessor()

    def test_preprocess_images(self) -> None:
        """
        Tests the preprocess_images method.

        Verifies that the method runs without errors and that preprocessed images are saved.

        Returns
        -------
        None
        """
        try:
            self.preprocessor.preprocess_images()
            assert os.path.exists(
                os.path.join(self.preprocessor.preprocessed_dir, "test_image.jpg")
            )
        except AssertionError:
            raise AssertionError("Preprocessed images were not saved correctly.")

    def test_resize_image(self) -> None:
        """
        Tests the resize_image method.

        Verifies that the method resizes an image to the specified size.

        Returns
        -------
        None
        """
        image = Image.new("RGB", (500, 500))
        resized_image = self.preprocessor.resize_image(image)
        assert resized_image.shape == self.preprocessor.image_size

    def test_normalize_image(self) -> None:
        """
        Tests the normalize_image method.

        Verifies that the method normalizes an image correctly.

        Returns
        -------
        None
        """
        image = np.ones((3, 500, 500))
        normalized_image = self.preprocessor.normalize_image(image)
        assert normalized_image.mean() == torch.tensor(0.0)

    def test_save_preprocessed_image(self) -> None:
        """
        Tests the save_preprocessed_image method.

        Verifies that the method saves an image to the correct directory.

        Returns
        -------
        None
        """
        image = torch.rand(3, 500, 500)
        saved_path = self.preprocessor.save_preprocessed_image(image, "test_image.jpg")
        assert saved_path == os.path.join(
            self.preprocessor.preprocessed_dir, "test_image.jpg"
        )


def run_integration_tests() -> None:
    """
    Runs the integration test suite for the ImagePreprocessor class.

    Instantiates the ImagePreprocessorIntegrationTest class and runs the test methods.

    Returns
    -------
    None
    """
    tests = ImagePreprocessorIntegrationTest()
    tests.setup_method()

    tests.test_preprocess_images()
    tests.test_resize_image()
    tests.test_normalize_image()
    tests.test_save_preprocessed_image()

    logger.info("Integration tests completed successfully.")


if __name__ == "__main__":
    run_integration_tests()