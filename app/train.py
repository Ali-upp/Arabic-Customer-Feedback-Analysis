from .model_utils import train_and_save


def main():
    pipeline, stats = train_and_save()
    print('Training completed. Stats:', stats)


if __name__ == '__main__':
    main()
