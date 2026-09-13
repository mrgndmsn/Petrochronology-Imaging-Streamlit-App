"""Explicit grain and spoke grouping without pooling unrelated identities."""
def parse_groups(text):
    groups={}
    for line in text.splitlines():
        if not line.strip(): continue
        if '=' not in line: raise ValueError('Use Group name = 1,2,3 on each line.')
        name, numbers=line.split('=',1)
        if not name.strip(): raise ValueError('Every group needs a name.')
        for number in numbers.split(','):
            try: number=int(number.strip())
            except ValueError: raise ValueError('Group members must be integer grain/profile numbers.')
            if number in groups: raise ValueError(f'Number {number} appears in more than one group.')
            groups[number]=name.strip()
    return groups

def apply_groups(frame, grains, profiles):
    frame=frame.copy()
    for column,text,label in [('grain_id',grains,'Grain'),('profile_number',profiles,'Profile')]:
        groups=parse_groups(text)
        frame[label+'_group']=frame[column].map(lambda n:groups.get(int(n),f'{label} {int(n)}'))
    frame['comparison_group']=frame.Grain_group+' / '+frame.Profile_group
    return frame
