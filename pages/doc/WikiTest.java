
import java.io.IOException;

import javax.xml.parsers.ParserConfigurationException;
import javax.xml.stream.XMLStreamException;
import javax.xml.transform.TransformerException;

import wiktionary_extractor.Page;
import wiktionary_extractor.PageExtractor;

public class WikiTest {

    public static void main(String[] args) throws IOException, XMLStreamException, ParserConfigurationException, TransformerException {
	String filename = "simplified_frwiktionary.xml.bz2";
	
	int count = 0;
	for (Page p : new PageExtractor(filename)) {
	    // Pour sauvergarder la page dans un fichier la page peut
	    // ensuite être rechargée en passant le nom du fichier à
	    // PageExtractor.
	    p.toXML("test.xml");
	    // Pour accéder au contenu (wikitext) et au titre d'une page
	    System.out.println(p.title);
	    System.out.println(p.content);
	    // Il y a **beaucoup** de pages. Sans condition d'arrêt le
	    // programme risque de mettre beaucoup de temps à
	    // s'exécuter
	    count += 1;
	    if (count == 10) {
		break;
	    }
	}
	System.out.println(count);
    }
    
}
