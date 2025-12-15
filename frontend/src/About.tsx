export default function About() {
  return (
    <div className="about-page">
      <div className="about-container">
        <h1 className="about-title">About UIT Chatbot</h1>
        
        <section className="about-section">
          <h2>Project Overview</h2>
          <p>
            The UIT Chatbot is an AI-powered assistant designed to help students and staff 
            quickly find information about the University of Information Technology's training 
            regulations, academic policies, and administrative procedures.
          </p>
          <p>
            Using advanced RAG (Retrieval-Augmented Generation) technology combined with 
            ontology-based knowledge graphs, this chatbot provides accurate, cited answers 
            to questions about UIT's regulations.
          </p>
        </section>

        <section className="about-section">
          <h2>Development Team</h2>
          <table className="team-table">
            <thead>
              <tr>
                <th>Name</th>
                <th>Class</th>
                <th>Student ID</th>
              </tr>
            </thead>
            <tbody>
              <tr>
                <td>Võ An Khôi</td>
                <td>KTPM2023.2</td>
                <td>23520790</td>
              </tr>
              <tr>
                <td>Võ Hồng Lương</td>
                <td>KTPM2023.2</td>
                <td>23520905</td>
              </tr>
              <tr>
                <td>Phạm Thị Kiều Diễm</td>
                <td>KTPM2023.1</td>
                <td>23520286</td>
              </tr>
            </tbody>
          </table>
        </section>

        <section className="about-section">
          <h2>Technologies Used</h2>
          <ul className="tech-list">
            <li>Natural Language Processing (NLP)</li>
            <li>Retrieval-Augmented Generation (RAG)</li>
            <li>Knowledge Graphs and Ontologies</li>
            <li>React + TypeScript</li>
            <li>FastAPI (Python)</li>
            <li>Docker</li>
          </ul>
        </section>

        <section className="about-section disclaimer">
          <h2>⚠️ Disclaimer</h2>
          <p>
            This project is developed for <strong>educational purposes only</strong> as part 
            of an academic assignment at the University of Information Technology (UIT).
          </p>
          <ul className="disclaimer-list">
            <li>
              The information provided by this chatbot is based on available UIT regulation 
              documents and may not always be up-to-date or complete.
            </li>
            <li>
              Users should always verify critical information with official UIT sources and 
              academic advisors.
            </li>
            <li>
              This is a prototype system and should not be used as the sole source for making 
              important academic decisions.
            </li>
            <li>
              The developers are not responsible for any decisions made based on the chatbot's 
              responses.
            </li>
            <li>
              All regulation documents and data belong to their respective copyright holders.
            </li>
          </ul>
          <div className="official-sources">
            <h3>For Official Information:</h3>
            <ul>
              <li>
                <a href="https://www.uit.edu.vn/" target="_blank" rel="noopener noreferrer">
                  UIT Official Website
                </a>
              </li>
              <li>UIT Student Affairs Office</li>
              <li>Your Academic Advisor</li>
            </ul>
          </div>
        </section>

        <section className="about-section">
          <h2>Academic Context</h2>
          <p>
            This project was developed as part of coursework at the University of Information 
            Technology, VNU-HCM. It demonstrates practical applications of modern AI and web 
            development technologies in solving real-world problems.
          </p>
        </section>

        <footer className="about-footer">
          <p>
            Developed by KTPM2023 students at University of Information Technology (UIT), VNU-HCM
          </p>
          <p className="year">© 2025</p>
        </footer>
      </div>
    </div>
  );
}
