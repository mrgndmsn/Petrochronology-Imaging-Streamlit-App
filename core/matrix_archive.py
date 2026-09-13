from zipfile import ZipFile

class MatrixMember:
    def __init__(self, archive, name):
        self.archive, self.name = archive, name
    def getvalue(self):
        return self.archive.read(self.name)


def matrix_archive_members(upload):
    upload.seek(0)
    archive=ZipFile(upload)
    names=[n for n in archive.namelist() if n.lower().endswith('.csv') and not n.startswith('__MACOSX/')]
    if not names:raise ValueError('ZIP contains no CSV matrices.')
    if len(names)!=len(set(names)):raise ValueError('ZIP contains duplicate member names.')
    return [MatrixMember(archive,n) for n in names]
