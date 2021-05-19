from setuptools import find_packages, setup
setup(
    name='EyeDiagnosisLib',
    packages=['GazeML_keras','GazeML_keras.detector','GazeML_keras.mtcnn_weights'],
    #packages=find_packages(include=['faceDetectCrop']),
    version='0.1.0',
    description='Diagnose Cataracts Library',
    author='Khaldoun Alnareiat',
    license='MIT',
    include_package_data=True
)