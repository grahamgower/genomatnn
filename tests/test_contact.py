import unittest

import msprime
import stdpopsim

import contact
import sim


def _5pop_test_demog(N=1000):
    populations = [stdpopsim.Population(f"pop{i}", f"Population {i}") for i in range(5)]
    pop_config = [
        msprime.PopulationConfiguration(
            initial_size=N, metadata=populations[i].asdict()
        )
        for i in range(len(populations))
    ]
    mig_mat = [
        [0, 0, 0, 0, 0],
        [0, 0, 1e-5, 0, 0],
        [0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0],
    ]
    dem_events = [
        msprime.MassMigration(time=100, source=0, destination=1, proportion=0.1),
        msprime.MassMigration(time=200, source=3, destination=2),
        msprime.MigrationRateChange(time=200, rate=0),
        msprime.MassMigration(time=300, source=1, destination=0),
        msprime.MassMigration(time=400, source=2, destination=4, proportion=0.1),
        msprime.MassMigration(time=600, source=2, destination=0),
        msprime.MassMigration(time=700, source=4, destination=0),
    ]
    return stdpopsim.DemographicModel(
        id="5pop_test",
        description="5pop_test",
        long_description="5pop_test",
        populations=populations,
        generation_time=1,
        population_configurations=pop_config,
        demographic_events=dem_events,
        migration_matrix=mig_mat,
    )


class TestContact(unittest.TestCase):
    def setUp(self):
        self.model = _5pop_test_demog()
        self.ddb = msprime.DemographyDebugger(
            demographic_events=self.model.demographic_events,
            population_configurations=self.model.population_configurations,
            migration_matrix=self.model.migration_matrix,
        )

    def test_tmcra(self):
        self.assertEqual(contact.tmrca(self.model, 0, 1), 100)
        self.assertEqual(contact.tmrca(self.model, 0, 2), 100)
        self.assertEqual(contact.tmrca(self.model, 1, 2), 0)
        self.assertEqual(contact.tmrca(self.model, 0, 3), 200)
        self.assertEqual(contact.tmrca(self.model, 1, 3), 200)
        self.assertEqual(contact.tmrca(self.model, 2, 3), 200)
        self.assertEqual(contact.tmrca(self.model, 0, 4), 400)
        self.assertEqual(contact.tmrca(self.model, 1, 4), 400)
        self.assertEqual(contact.tmrca(self.model, 2, 4), 400)
        self.assertEqual(contact.tmrca(self.model, 3, 4), 400)

    def test_min_coalescent_time_ancient_samples(self):
        self.assertEqual(
            contact.min_coalescence_time(
                self.ddb,
                [
                    msprime.Sample(time=150, population=0),
                    msprime.Sample(time=0, population=1),
                ],
            ),
            300,
        )
        self.assertEqual(
            contact.min_coalescence_time(
                self.ddb,
                [
                    msprime.Sample(time=150, population=0),
                    msprime.Sample(time=0, population=4),
                ],
            ),
            700,
        )
        self.assertEqual(
            contact.min_coalescence_time(
                self.ddb,
                [
                    msprime.Sample(time=0, population=0),
                    msprime.Sample(time=450, population=4),
                ],
            ),
            450,
        )

    def test_split_time(self):
        model = sim.hominin_composite()
        pop = {p.id: i for i, p in enumerate(model.populations)}
        self.assertEqual(
            contact.split_time(model, pop["Nea"], pop["CEU"]),
            550e3 / model.generation_time,
        )
        self.assertEqual(
            contact.split_time(model, pop["YRI"], pop["CEU"]),
            65.7e3 / model.generation_time,
        )

        species = stdpopsim.get_species("HomSap")
        model = species.get_demographic_model("PapuansOutOfAfrica_10J19")
        pop = {p.id: i for i, p in enumerate(model.populations)}

        self.assertEqual(contact.split_time(model, pop["Papuan"], pop["Ghost"]), 1784)
        self.assertEqual(contact.split_time(model, pop["DenA"], pop["NeaA"]), 15090)
        self.assertEqual(contact.split_time(model, pop["DenA"], pop["Den1"]), 9750)
        self.assertEqual(contact.split_time(model, pop["DenA"], pop["Den2"]), 12500)
        self.assertEqual(contact.split_time(model, pop["CEU"], pop["CHB"]), 1293)
