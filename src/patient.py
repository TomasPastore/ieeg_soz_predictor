class Patient():
    def __init__(self, id, age, gender=None, electrodes=None):
        if electrodes is None:
            electrodes = []
        self.id = id
        self.age = age
        self.gender = gender
        self.electrodes = electrodes

    def get_types_count(self, types):
        res = 0
        for e in self.electrodes:
            res += e.get_events_count(types)
        return res

    def add_electrode(self, electrode):
        self.electrodes.append(electrode)

    def remove_electrodes(self, to_remove):
        for e in to_remove:
            self.electrodes.remove(e)

    def electrode_names(self):
        return [e.name for e in self.electrodes]

    def get_electrode(self, e_name):
        try:
            return [e for e in self.electrodes if e.name == e_name][0]
        except IndexError:
            raise IndexError('Electrode not present in patient')

    def print(self):
        print('Printing patient {0}: '.format(self.id))
        print('\tAge: {0}'.format(self.age))
        print('\tElectrodes: ------------------------------')
        for e in self.electrodes:
            e.print()
        print('------------------------------------------')

    # Esta sirve para hacer particiones balanceadas,
    # minimizo el desbalanceo entre grupos de pacientes.
    # Si hago que los sets de test esten balanceados en clases hace que el
    # entrenamiento este desbalanceado y prediga mal. Es mejor balancear
    # entrenamiento. Mejor no usar.
    def get_classes_weight(self, hfo_type_name):
        negative_class_count = 0
        positive_class_count = 0
        tot_count = 0
        for e in self.electrodes:
            for h in e.events[hfo_type_name]:
                tot_count += 1
                if h.info['soz']:
                    positive_class_count += 1
                else:
                    negative_class_count += 1
        return negative_class_count, positive_class_count

    # Determines if the patient has any electrode in loc
    def has_elec_in(self, loc):
        from db_parsing import get_granularity
        return any([getattr(e, 'loc{i}'.format(i=get_granularity(loc)))
                    == loc for e in self.electrodes])

    # Returns true iff self has soz electrode in loc_name
    def has_epilepsy_in_loc(self, loc_name):
        from db_parsing import get_granularity

        if loc_name == 'Whole Brain':  # Has epilepsy in any part of the brain
            return any([e.soz for e in
                        self.electrodes])  # assumes that e.soz is parsed
        else:  # looks if any soz electrode matches its loc_name in the
            # correct granularity tags
            granularity = get_granularity(loc_name)
            return any([e.soz and getattr(e, e.loc_field_by_granularity(
                granularity)) == loc_name for e in self.electrodes])

    def has_epileptic_activity(self, loc_name, balance_types):
        granularity = get_granularity(loc_name)
        return any([(loc_name == 'Whole Brain' or
                     getattr(e, e.loc_field_by_granularity(
                         granularity)) == loc_name) and
                    (e.has_epileptic_activity(balance_types)) for e in
                    self.electrodes])

    def has_nsoz_activity(self, loc_name, balance_types):
        granularity = get_granularity(loc_name)
        return any([(loc_name == 'Whole Brain' or
                     getattr(e, e.loc_field_by_granularity(
                         granularity)) == loc_name) and
                    (e.has_nsoz_activity(balance_types)) for e in
                    self.electrodes])

    def has_epilepsy_in_all_locs(self, locations):
        return all(
            [self.has_epilepsy_in_loc(location) for location in locations])

    # Iff all electrodes are loc field is inside the list allowed, given as
    # parameter locations
    # "empty" in locations allows null elements
    def has_epilepsy_restricted_to(self, granularity, locations):
        if granularity == 0:
            return True
        else:
            restricted = True
            for e in self.electrodes:
                if e.soz and getattr(e, e.loc_field_by_granularity(
                        granularity)) not in locations:
                    restricted = False
                    break
            return restricted

    def print_age_gender(self):
        print('Patient: ', self.id)
        print('AGE: ', self.age)
        print('GENDER: ', self.gender)


def get_granularity(loc):
    if loc is None or loc == 'Whole Brain':
        return 0
    for i in range(6):
        if loc in all_loc_names(i):
            return i
    raise RuntimeError('Unknown location name: {0}'.format(loc))


def all_loc_names(granularity):  # all to test
    if granularity == 0:
        return ['Whole Brain']
    if granularity == 2:
        return ['Frontal Lobe', 'Temporal Lobe', 'Parietal Lobe',
                'Limbic Lobe', 'Occipital Lobe']
    elif granularity == 3:
        # removed: 'Extra-Nuclear',
        return ['Angular Gyrus', 'Anterior Cingulate', 'Caudate',
                'Cerebellar Tonsil', 'Cingulate Gyrus', 'Claustrum', 'Culmen',
                'Cuneus', 'Declive', 'Fusiform Gyrus',
                'Inferior Frontal Gyrus', 'Inferior Occipital Gyrus',
                'Inferior Parietal Lobule', 'Inferior Temporal Gyrus', 'Insula',
                'Lentiform Nucleus', 'Lingual Gyrus', 'Medial Frontal Gyrus',
                'Middle Frontal Gyrus', 'Middle Occipital Gyrus',
                'Middle Temporal Gyrus', 'Orbital Gyrus', 'Paracentral Lobule',
                'Parahippocampal Gyrus', 'Postcentral Gyrus',
                'Posterior Cingulate', 'Precentral Gyrus', 'Precuneus',
                'Pyramis', 'Sub-Gyral', 'Subcallosal Gyrus',
                'Superior Frontal Gyrus', 'Superior Occipital Gyrus',
                'Superior Parietal Lobule', 'Superior Temporal Gyrus',
                'Supramarginal Gyrus', 'Thalamus', 'Transverse Temporal Gyrus',
                'Tuber', 'Uncus']

    elif granularity == 5:
        return ['Amygdala', 'Brodmann area 1', 'Brodmann area 10',
                'Brodmann area 11', 'Brodmann area 13', 'Brodmann area 17',
                'Brodmann area 18', 'Brodmann area 19', 'Brodmann area 2',
                'Brodmann area 20', 'Brodmann area 21', 'Brodmann area 22',
                'Brodmann area 23', 'Brodmann area 24', 'Brodmann area 25',
                'Brodmann area 27', 'Brodmann area 28', 'Brodmann area 29',
                'Brodmann area 3', 'Brodmann area 30', 'Brodmann area 31',
                'Brodmann area 32', 'Brodmann area 33', 'Brodmann area 34',
                'Brodmann area 35', 'Brodmann area 36', 'Brodmann area 37',
                'Brodmann area 37 ', 'Brodmann area 38', 'Brodmann area 39',
                'Brodmann area 4', 'Brodmann area 40', 'Brodmann area 41',
                'Brodmann area 42', 'Brodmann area 44', 'Brodmann area 45',
                'Brodmann area 46', 'Brodmann area 47', 'Brodmann area 5',
                'Brodmann area 6', 'Brodmann area 7', 'Brodmann area 8',
                'Brodmann area 9', 'Caudate Body', 'Caudate Head',
                'Caudate Tail', 'Hippocampus', 'Lateral Globus Pallidus',
                'Pulvinar', 'Putamen']
    else:
        return []
