from app.DB_access import DatabaseMongo


def reserver_salle(infos_reservation):
    db = DatabaseMongo()
    reservations_collection = db.get_collection("reservation")
    salle_collection = db.get_collection("salle")
    salle_doc = salle_collection.find_one({"nom": infos_reservation.get("salle")})
    infos_reservation = {
        "utilisateur_id": infos_reservation.get("utilisateur_id"),
        "salle": salle_doc.get("_id"),
        "creneau": {
            "jour": infos_reservation.get("creneau", {}).get("jour"),
            "heure_debut": infos_reservation.get("creneau", {}).get("heure_debut"),
            "heure_fin": infos_reservation.get("creneau", {}).get("heure_fin"),
        }
    }
    # Insérer les informations de réservation dans la collection
    result = reservations_collection.insert_one(infos_reservation)
    
    db.close()
    return result.inserted_id

