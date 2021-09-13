import boto3.session
from multiprocessing import current_process

"""specifying desired keys for uploading/downloading files to s3"""
prefixes = ['raw/table-bank/detection-data/Latex/images/',
            'raw/table-bank/detection-data/Word/images/']
destination_prefixes = ['intermediate/table-bank/ocr-data/latex/',
                        'intermediate/table-bank/ocr-data/word/']

url_files = ['url_latex.csv',
             'url_word.csv']


def get_bucket():
    """getting mfa code from authentication device"""
    mfa_otp = input("Enter the MFA code for " + current_process().name + ": ")

    """getting temporary credentials"""
    client = boto3.client('sts')
    response = client.get_session_token(DurationSeconds=129600,
                                        SerialNumber='arn:aws:iam::536930143272:mfa/billy.ohara',
                                        TokenCode=mfa_otp)

    """starting session (max 36 hours)"""
    my_session = boto3.session.Session(aws_access_key_id=response['Credentials']['AccessKeyId'],
                                       aws_secret_access_key=response['Credentials']['SecretAccessKey'],
                                       aws_session_token=response['Credentials']['SessionToken'])

    """connecting to s3 table identification bucket"""
    s3 = my_session.resource('s3')
    bucket = s3.Bucket('table-identification')
    return bucket


def counter(bucket, prefix):
    """counts number of objects in a folder"""
    objs = bucket.objects.filter(Prefix=prefix)
    count = 0
    for obj in objs:
        count += 1
        if count % 10000 == 0:
            print(count)
    print('finished with ' + str(count))

