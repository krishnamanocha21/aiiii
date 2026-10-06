# aiiii
here i learn the ai 

# NOTE
 st.session_state.message and the st.session_state.history are the same things.

# FOR RAG COMPLETE CHATBOT THING
 1. we built the agent inside the process document because it depends completly on the document retrival context .

 2. we store the agent as it is a local variable in the process document which get finished once done 
 so we have to store it somewhere

 3. with open(path + file.name, "wb") as f:  f.write(file.getvalue())
 this open the file path and write binary as an alias as (f) and if not exist it will create one 

 4. 