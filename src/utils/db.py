from neo4j import GraphDatabase
from src.utils.config import NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD


class GraphDB:
    """
    Thin wrapper around the Neo4j driver.

    Usage:
        db = GraphDB()
        db.create_worker_node(...)
        db.close()

    Make sure Neo4j is running locally and credentials in config.py are correct
    before calling any method.
    """

    def __init__(self):
        self.driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))

    def close(self):
        self.driver.close()

    def clear_graph(self):
        with self.driver.session() as session:
            session.run("MATCH (n) DETACH DELETE n")

    def create_worker_node(self, worker_id, label, count, risk, district_id):
        query = """
        MERGE (w:Worker {id: $worker_id})
        SET w.label        = $label,
            w.estimated_count = $count,
            w.risk_score   = $risk,
            w.district_id  = $district_id
        """
        with self.driver.session() as session:
            session.run(query, worker_id=worker_id, label=label,
                        count=count, risk=risk, district_id=district_id)

    def create_platform_node(self, platform_id, name, exit_risk):
        query = """
        MERGE (p:Platform {id: $platform_id})
        SET p.name = $name, p.exit_risk = $exit_risk
        """
        with self.driver.session() as session:
            session.run(query, platform_id=platform_id, name=name, exit_risk=exit_risk)

    def create_skill_node(self, skill_id, name, automation_risk):
        query = """
        MERGE (s:Skill {id: $skill_id})
        SET s.name = $name, s.automation_risk = $automation_risk
        """
        with self.driver.session() as session:
            session.run(query, skill_id=skill_id, name=name, automation_risk=automation_risk)

    def create_district_node(self, district_id, name, state):
        query = """
        MERGE (d:District {id: $district_id})
        SET d.name = $name, d.state = $state
        """
        with self.driver.session() as session:
            session.run(query, district_id=district_id, name=name, state=state)

    def create_edge(self, source_id, target_id, edge_type, weight=1.0):
        query = f"""
        MATCH (a {{id: $source_id}}), (b {{id: $target_id}})
        MERGE (a)-[r:{edge_type}]->(b)
        SET r.weight = $weight
        """
        with self.driver.session() as session:
            session.run(query, source_id=source_id, target_id=target_id, weight=weight)

    def get_worker_risk_scores(self):
        query = "MATCH (w:Worker) RETURN w.id AS id, w.risk_score AS risk ORDER BY risk DESC"
        with self.driver.session() as session:
            result = session.run(query)
            return [dict(record) for record in result]

    def get_platform_worker_count(self, platform_id):
        query = """
        MATCH (w:Worker)-[:WORKS_ON]->(p:Platform {id: $platform_id})
        RETURN count(w) AS worker_count
        """
        with self.driver.session() as session:
            result = session.run(query, platform_id=platform_id)
            return result.single()["worker_count"]
