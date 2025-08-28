# This script prepares transcriptions from the National Archives into a format usable by olmOCR
# What it will do is take in a path which will contain a folder structure of either collections or record groups from the NA
# Inside each of those folders, it will go and read every jsonl file and check each record
# {
#     "record": {
#         "accessRestriction": {
#             "status": "Unrestricted"
#         },
# ....
# So, first we check to see that the record.accessRestriction.status is Unrestricted
# Next, we go look for the digitalObjects section
# "digitalObjects": [
#     {
#         "objectFileSize": 12368728,
#         "objectFilename": "23857158-001-068-0001.tif",
#         "objectId": "310993715",
#         "objectType": "Image (TIFF)",
#         "objectUrl": "https://s3.amazonaws.com/NARAprodstorage/lz/dc-metro/rg-341/23857158/23857158-001-068/23857158-001-068-0001.tif"
#     },
#     {
#         "objectFileSize": 9496446,
#         "objectFilename": "23857158-001-068-0002.tif",
#         "objectId": "310993716",
#         "objectType": "Image (TIFF)",
#         "objectUrl": "https://s3.amazonaws.com/NARAprodstorage/lz/dc-metro/rg-341/23857158/23857158-001-068/23857158-001-068-0002.tif"
#     }, ...
# If they are images, we download them and move onto to the next phase
# Where we look at record_transcription tags...
# "record_transcription": [
#         {
#             "contribution": "This is the transcription",
#             "contributionId": "b1200268-0802-3e96-950e-86cb490af7a5",
#             "contributionSequence": 2,
#             "contributionType": "transcription",
#             "contributors": [
#                 {
#                     "contributionSequence": 1,
#                     "createdAt": "2018-09-07 22:03:02",
#                     "fullName": "Cody Jones",
#                     "naraStaff": false,
#                     "userId": "dff3eed0-38e5-35fc-b7e7-d2d58b023262",
#                     "userName": "Avogadro"
#                 },
#                 {
#                     "contributionSequence": 2,
#                     "createdAt": "2018-09-07 22:05:53",
#                     "fullName": "Cody Jones",
#                     "naraStaff": false,
#                     "userId": "dff3eed0-38e5-35fc-b7e7-d2d58b023262",
#                     "userName": "Avogadro"
#                 }
#             ],
#             "createdAt": "2018-09-07 22:05:53",
#             "parentContributionId": "01c9fab3-8d1e-3027-96f9-890728825f63",
#             "recordType": "contribution",
#             "target": {
#                 "naId": 75718510,
#                 "objectId": "75718511",
#                 "pageNum": 1
#             }
#         }
# We also check the  record tag to make sure aiMachineGenerated is false
# "record_tag": [
#         {
#             "aiMachineGenerated": false,
#             "contribution": "uap-tx-2023",
#             "contributionId": "2f3e9a6e-cfb9-4823-8251-a0f2d129b9e2",
#             "contributionType": "tag",
#             "contributor": {
#                 "fullName": "Erica Boudreau",
#                 "naraStaff": true,
#                 "userId": "8882c6b7-0906-3298-916b-d35132a528be",
#                 "userName": "NARADescriptionProgramStaff"
#             },
#             "createdAt": "2024-12-16 16:47:12",
#             "recordType": "contribution",
#             "source": "naraStaff",
#             "target": {
#                 "naId": 310993714
#             }
#         },
# Then, for each image, which is typically a scanned document page, we create a dataset in olmocr-format, where you have a .md file and a .pdf file named with the ItemID in a folder structure for 
# each initial jsonl file. Ex if you had rg_341/rg_341-53.jsonl, then you'd make rg_341/object_id.md and rg_341/object_id.pdf
# If you have a TIFF file, you can compress it to jpg at 98% quality, targetting around 1-2MB in size.
# Then use https://pypi.org/project/img2pdf/ to convert the images to PDFs losslessly.