from pathlib import Path
import boto3
from PIL import Image, ImageDraw, ImageFont
from mypy_boto3_rekognition.type_defs import CelebrityTypeDef, RecognizeCelebritiesResponseTypeDef

client = boto3.client('rekognition')

def get_path(file_name: str) -> str:
    return str(Path(__file__).parent / 'images' / file_name)

def get_images_from_folder(folder_path: str) -> list[str]:
    return [str(p) for p in Path(folder_path).glob('*.jpg')]

def recognize_celebrities(image_path: str) -> RecognizeCelebritiesResponseTypeDef:
    with open(image_path, 'rb') as image:
        response = client.recognize_celebrities(Image={'Bytes': image.read()})
    return response

def draw_boxes(image_path: str, output_path: str, celebrities: list[CelebrityTypeDef]):
    image = Image.open(image_path)
    draw = ImageDraw.Draw(image)
    font = ImageFont.truetype('Ubuntu-R.ttf', 20)

    width, height = image.size

    for celebrity in celebrities:
        box = celebrity['Face']['BoundingBox'] # type: ignore
        left = int(box['Left'] * width) # type: ignore
        top = int(box['Top'] * height) # type: ignore
        right = int((box['Left'] + box['Width']) * width) # type: ignore
        bottom = int((box['Top'] + box['Height']) * height) # type: ignore

        if celebrity['MatchConfidence'] > 90:
            draw.rectangle([left, top, right, bottom], outline='red', width=3)

            name = celebrity['Name'] # type: ignore
            position = (left, top - 20)
            bbox = draw.textbbox(position, name, font=font)
            draw.rectangle(bbox, fill='red')
            draw.text(position, name, font=font, fill='white')

    image.save(output_path)
    print(f'Image saved to {output_path}')

if __name__ == '__main__':
    folder_path = str(Path(__file__).parent / 'images')
    image_paths = get_images_from_folder(folder_path)

    for image_path in image_paths:
        output_image_path = str(Path(image_path).with_name(f'resultado_{Path(image_path).name}'))
        response = recognize_celebrities(image_path)

        if response['CelebrityFaces']:
            print(f'Celebrities recognized in {image_path}:')
            for celebrity in response['CelebrityFaces']:
                print(f"Name: {celebrity['Name']}, Confidence: {celebrity['MatchConfidence']:.2f}")

            draw_boxes(image_path, output_image_path, response['CelebrityFaces'])
        else:
            print(f'No celebrities recognized in {image_path}')
