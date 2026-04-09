import json

class is_visible_toggle:
    def __init__(self, assembly_file):
        self.assembly_file = assembly_file

    def toggle_visibility(self, body_ids_to_toggle):
        """
        Deactivate specific parts of an assembly object and write the changes to its config.
        """
        with open(self.assembly_file, 'r') as file:
            data = json.load(file)

        if "root" in data and "bodies" in data["root"]:
            bodies = data["root"].get("bodies", {})
            changes_made = False

            for body_id, body in bodies.items():
                if body_id in body_ids_to_toggle:
                    body["is_visible"] = False
                    changes_made = True
                    print(f"Toggled visibility for body ID: {body_id} to {body['is_visible']}")        

        if "occurrences" in data:
            occurrences = data.get("occurrences", {})
            changes_made = False

            for occ in occurrences.values():
                bodies = occ.get("bodies", {})
                for body_id, body in bodies.items():
                    if body_id in body_ids_to_toggle:
                        body["is_visible"] = False
                        changes_made = True
                        print(f"Toggled visibility for body ID: {body_id} to {body['is_visible']}")


        else:
            print("Unknown data structure format.")
            return

        if changes_made:
            with open(self.assembly_file, 'w') as file:
                json.dump(data, file, indent=4)
            
    def get_node_visibility(self):
        """
        Get the count of visible parts and a dict[int, str] mapping their index to all visible ids.
        """
        with open(self.assembly_file, 'r') as file:
            data = json.load(file)
        
        visible_count = 0
        visible_ids = {}


        if "root" in data and "bodies" in data["root"]:
            bodies = data["root"].get("bodies", {})
            for body_id, body in bodies.items():
                if body.get("is_visible", False):
                    visible_ids[visible_count] = body_id
                    visible_count += 1

        if "occurrences" in data:
            occurrences = data.get("occurrences", {})
            for occ_id, occ in occurrences.items():
                bodies = occ.get("bodies", {})
                for body_id, body in bodies.items():
                    if body.get("is_visible", False):
                        visible_ids[visible_count] = body_id
                        visible_count += 1

        else:
            print("Unknown data structure format.")
            return visible_count, visible_ids

        return visible_count, visible_ids
    
    def reset_visibility(self):
        """
        Activate all parts of an assembly object and write the changes to its config.
        """
        with open(self.assembly_file, 'r') as file:
            data = json.load(file)
        
        changes_made = False

        if "root" in data and "bodies" in data["root"]:
            bodies = data["root"].get("bodies", {})
            for body_id, body in bodies.items():
                if "is_visible" in body and not body["is_visible"]:
                    body["is_visible"] = True
                    changes_made = True

        if "occurrences" in data:
            occurrences = data.get("occurrences", {})
            for occ_id, occ in occurrences.items():
                if not occ.get("is_visible", True):
                    occ["is_visible"] = True
                    changes_made = True
                
                bodies = occ.get("bodies", {})
                for body_id, body in bodies.items():
                    if not body.get("is_visible", True):
                        body["is_visible"] = True
                        changes_made = True
                        

        else:
            print("Unknown data structure format.")
            return

        if changes_made:
            with open(self.assembly_file, 'w') as file:
                json.dump(data, file, indent=4)
